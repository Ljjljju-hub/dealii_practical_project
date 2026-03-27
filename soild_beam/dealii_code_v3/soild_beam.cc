#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
// 这个头文件本身不写任何数学算法。它是一个纯粹的“接口别名工厂（Alias Factory）”。
#include <deal.II/lac/generic_linear_algebra.h>
// 强制使用Tilinos
#define FORCE_USE_OF_TRILINOS
/*
它在底层把 PETSc 和 Trilinos 那千差万别的类名，强行打包成了两个名字和调用方式 100% 相同的标准化命名空间：
dealii::LinearAlgebraPETSc
dealii::LinearAlgebraTrilinos
不管底层用的是谁，在这个头文件的包装下，分布式的向量统统叫 MPI::Vector，分布式的矩阵统统叫 MPI::SparseMatrix，求解器统统叫 SolverCG。
*/
// 配合 namespace LA 施展魔法

  namespace LA
  {
  #if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
    !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace dealii::LinearAlgebraPETSc;
  #  define USE_PETSC_LA
  #elif defined(DEAL_II_WITH_TRILINOS)
    using namespace dealii::LinearAlgebraTrilinos;
  #else
  #  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
  #endif
  } // namespace LA
// 在这段代码之后，你在 SolidBeam 类里定义矩阵时，只需要写：
// LA::MPI::SparseMatrix system_matrix;

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
// 被那个万能的 generic_linear_algebra.h 默默地承包了
// #include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
// 进行了拆分，精确导入
// #include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_cg.h>
// #include <deal.II/lac/generic_linear_algebra.h>替代了，LA::MPI::PreconditionAMG
// #include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
//  专为 p4est 分布式网格量身定制的救命神器！
#include <deal.II/lac/sparsity_tools.h>

// #include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
// #include <deal.II/grid/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

// <mpi.h>也不需要了
// #include <deal.II/base/mpi.h>
#include <deal.II/base/index_set.h>
// 不需要GridTools::partition_triangulation，在底层召唤 METIS 或 Zoltan 图划分算法，
// 像涂色一样给每个单元强行打上 subdomain_id 的标签。
// #include <deal.II/grid/grid_tools.h>

// 现在的 DoFHandler 变得极其聪明。当它发现自己绑定的是一个纯分布式网格时，
//你在调用 dof_handler.distribute_dofs(fe); 的那一瞬间，它底层就会自动按照子域进行连续编号
// #include <deal.II/dofs/dof_renumbering.h>


#include <fstream>
#include <iostream>
#include <string>
// 将标准输出 std::cout 替换为一个新的流 pcout，该流在并行计算中用于仅在其中一个 MPI 进程上生成输出。
#include <deal.II/base/conditional_ostream.h>

// 只要你装了 CUDA Toolkit / Nsight，这个头文件默认就在系统里
#include <nvtx3/nvtx3.hpp>

using namespace std;
using namespace dealii;

template <int dim>
class SoildBeam{
public:
    SoildBeam(const std::string inp_path);
    void run();

private:
    void setup_boundary_ids();
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    // 【新增】MPI 通信器与自由度归属账本
    MPI_Comm mpi_communicator;
    IndexSet locally_owned_dofs;    // 属于我的自由度
    IndexSet locally_relevant_dofs; // 我能看见的自由度（含幽灵节点）

    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    parallel::distributed::Triangulation<dim> triangulation;    // 并行网格
    DoFHandler<dim> dof_handler;    // 不变

    const FESystem<dim> fe;

    AffineConstraints<double> constraints;  // 不变

    // 【修改】原生的 SparsityPattern 和 SparseMatrix 替换为 Trilinos
    // TrilinosWrappers::SparseMatrix system_matrix;

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector locally_relevant_solution;
    LA::MPI::Vector system_rhs;

    // inp文件路径
    std::string inp_path;

    // 1. 先声明文件流（必须在计时器前面！）
    std::ofstream timer_file;


    // 【新增】高级性能剖析计时器
    // output_results声明为const
    // 在output_results内是需要修改TimerOutput的，添加mutable
    // mpi中TimerOutput记录时间的对象，传入的是普通的“std::ofstream timer_file;”
    // 但是可以正确在mpi环境下只输出一次内容.
    mutable TimerOutput computing_timer;

};  // end of Class SoildBeam

template <int dim>
void right_hand_side(const std::vector<Point<dim>> &points, 
    std::vector<Tensor<1, dim>> &values){

    AssertDimension(values.size(), points.size());
    Assert(dim >= 2, ExcNotImplemented());

}

// 注意这里要把 mpi_communicator 传给计时器，这样多个核心的时间才会自动统计。
template <int dim>
SoildBeam<dim>::SoildBeam(const std::string inp_path)
:mpi_communicator(MPI_COMM_WORLD),  // 初始化全局通信域
n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
triangulation(
    mpi_communicator, 
    // Triangulation<dim> 是最基础的父类（基类 / Base Class）
    // 通过parallel::distributed::Triangulation<dim>引入
    // 当你定义一个子类（派生类）时，编译器必须、立刻、马上看到它父类（基类）的全部完整定义！
    typename Triangulation<dim>::MeshSmoothing(
        Triangulation<dim>::smoothing_on_refinement |
        Triangulation<dim>::smoothing_on_coarsening
    )
),
dof_handler(triangulation), fe(FE_Q<dim>(1) ^ dim), inp_path(inp_path),
// timer_file("timer_summary.txt"), 先不进行初始化
computing_timer(mpi_communicator, timer_file, TimerOutput::summary, TimerOutput::wall_times){
    // 【核心修复】只允许 0 号进程真正去触碰硬盘，打开文件
    if (this_mpi_process == 0) {
        timer_file.open("timer_summary.txt");
    }
}

// 搜索位移边界的节点，为位移边界节点设置id
// 在初始只执行一次
// dealii中细化的面的网格id会继承
template<int dim>
void SoildBeam<dim>::setup_boundary_ids(){
    TimerOutput::Scope t(computing_timer, "0. setup_boundary_ids");
    // 遍历网格中所有处于激活状态的单元
    for(auto &cell : triangulation.active_cell_iterators()){
        // 遍历当前单元的所有面 (三维六面体有6个面)
        for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f){
            // 判断这个面是否在整个几何体的外边界上（剔除单元之间的内部面）
            if(cell->face(f)->at_boundary()){
                // 获取这个面的几何中心坐标
                const Point<dim> face_center = cell->face(f)->center();

                // 判断这个面的中心 X 坐标是否等于 0 (使用 1e-6 作为浮点数容差)
                if(std::abs(face_center[0] - 0.) < 1e-6){
                    // 如果是，就把这个面的 boundary_id 改为 1
                    cell->face(f)->set_boundary_id(1);
                }

                // 添加面力
                // 如果你还有一个受力面在 Y=10，也可以顺便标记
                if (std::abs(face_center[1] - 10.0) < 1e-6)
                {
                    cell->face(f)->set_boundary_id(2);
                }
            }
        }
    }
}


template <int dim>
void SoildBeam<dim>::setup_system(){
    TimerOutput::Scope t(computing_timer, "1. setup_system");
    // 不需要了
    // // 【核心 1】呼叫 METIS 将全局网格打上分区标签
    // GridTools::partition_triangulation(n_mpi_processes, triangulation);

    // 爆发极其剧烈的 MPI 通信！
    dof_handler.distribute_dofs(fe);
    // 不需要了
    // 对域从新调整编号
    // DoFRenumbering::subdomain_wise(dof_handler);

    if(this_mpi_process == 0){
        deallog << "   Number of active cells:       "
                << triangulation.n_global_active_cells() << std::endl
                << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                << std::endl;
    }

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    // 完全不通信（0 通信），它纯粹是在本地内存里“查字典”！
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
    // 多分配一圈内存空间
    // 因为求解完方程后，你需要做**“读取（Read）”**操作（比如输出 .vtu 结果，或者计算单元应变）。
    // 需要存储幽灵节点上的值
    locally_relevant_solution.reinit(
        locally_owned_dofs,
        locally_relevant_dofs,
        mpi_communicator
    );
    system_rhs.reinit(
        locally_owned_dofs,
        mpi_communicator
    );

    constraints.clear();
    // 新增需要传入两个dofs
    // 问题：目前不知道constraints是怎么实现位移边界和悬挂节点约束的。与自由度、幽灵节点的关系是什么。
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
        dof_handler, types::boundary_id(1), Functions::ZeroFunction<dim>(dim), constraints
    );
    constraints.close();

    // // 阶段一的“内存杀手”：全局稀疏蓝图
    // DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    // DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    // // 交接给 Trilinos：化虚为实
    // const std::vector<IndexSet> locally_owned_dofs_per_proc = 
    //     DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
    // const IndexSet &locally_owned_dofs = 
    //     locally_owned_dofs_per_proc[this_mpi_process];

    // system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    // 触发一次剧烈的 MPI 通信
    // Rank 0 的本地视角：Rank 0 处理这个悬挂节点时，发现 $A$ 和 $B$ 产生了数学上的耦合。
    // 于是，Rank 0 在自己的本地 dsp 蓝图上，给 $(A, B)$ 和 $(B, A)$ 都画了圈。
    // Rank 1 的本地视角：Rank 1 的领地里，根本没有包含这个悬挂节点的物理单元！
    // 因为悬挂节点两侧单元不一样，悬挂节点不是幽灵节点
    SparsityTools::distribute_sparsity_pattern(
        dsp,
        dof_handler.locally_owned_dofs(),   
        mpi_communicator,
        locally_relevant_dofs
    );

    system_matrix.reinit(
        locally_owned_dofs,
        locally_owned_dofs,
        dsp,
        mpi_communicator
    );

}


template<int dim>
void SoildBeam<dim>::assemble_system(){
    // 【新增】哨兵：它会自动记录整个 assemble_system 函数的耗时，标签名为 "2. Assemble system"
    TimerOutput::Scope t(computing_timer, "2. assemble_system");

    // 只要加上这一行，Nsight 的时间轴上就会出现一个写着 "Assemble_System" 的完美色块！
    nvtx3::scoped_range marker{"Assemble_System"};

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(
        fe,
        quadrature_formula,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values
    );

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    // 【新增】面积分工具！注意传入的是 face_quadrature_formula (降一维的积分公式)
    QGauss<dim-1> face_quadraqure_formula(fe.degree + 1);
    FEFaceValues<dim> fe_face_values(
        fe,
        face_quadraqure_formula,
        update_values | update_quadrature_points |
        update_normal_vectors | update_JxW_values
    );

    unsigned int n_face_q_points = face_quadraqure_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);

    // 定义材料参数
    const double E = 210e9;     // 弹性模量 (例如：2.1 GPa)
    const double nu = 0.3;    // 泊松比

    // 2. 转换为拉梅系数 (Lamé parameters)
    const double lambda_val = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double mu_val     = E / (2.0 * (1.0 + nu));

    Functions::ConstantFunction<dim> lambda(lambda_val), mu(mu_val);

    std::vector<Tensor<1, dim>> rhs_values(n_q_points);


    // 对所有单元进行遍历
    for(const auto &cell : dof_handler.active_cell_iterators()){
        // ==========================================
        // 【核心修改】只处理属于当前进程的单元！
        // ==========================================
        // if(cell->subdomain_id() != this_mpi_process){
        //     continue;
        // }
        if(!(cell->is_locally_owned())){
            continue;
        }

        fe_values.reinit(cell);

        cell_matrix = 0;
        cell_rhs = 0;

        lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
        mu.value_list(fe_values.get_quadrature_points(), mu_values);
        right_hand_side(fe_values.get_quadrature_points(), rhs_values);

        for(const unsigned int i : fe_values.dof_indices()){
            const unsigned int component_i = 
                fe.system_to_component_index(i).first;
            
                for(const unsigned int j : fe_values.dof_indices()){
                    const unsigned int component_j = 
                        fe.system_to_component_index(j).first;

                    for(const unsigned int q_point : fe_values.quadrature_point_indices()){
                        cell_matrix(i, j) += (
                            (
                                fe_values.shape_grad(i, q_point)[component_i] *
                                fe_values.shape_grad(j, q_point)[component_j] *
                                lambda_values[q_point]
                            ) +
                            (
                                fe_values.shape_grad(i, q_point)[component_j] *
                                fe_values.shape_grad(j, q_point)[component_i] *
                                mu_values[q_point]
                            ) +
                            (
                                (component_i == component_j) ?
                                (
                                    fe_values.shape_grad(i, q_point) *
                                    fe_values.shape_grad(j, q_point) *
                                    mu_values[q_point]
                                ) : 0
                            )
                        ) * fe_values.JxW(q_point);
                    }
                }
        }

        // 右端荷载项
        for(const unsigned int i : fe_values.dof_indices()){
            const unsigned int component_i = 
            fe.system_to_component_index(i).first;

            for(const unsigned int q_point : fe_values.quadrature_point_indices()){
                cell_rhs(i) += fe_values.shape_value(i, q_point) *
                rhs_values[q_point][component_i] * fe_values.JxW(q_point);
            }
        }

        // 右端力边界项
        // ==========================================
        // 【新增】面积分：处理面力 (Neumann 边界条件)
        // ==========================================
        // 遍历当前单元的所有面
        for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f){
            if(cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 2){
                fe_face_values.reinit(cell, f);

                // 遍历面上的高斯积分点
                for(unsigned int q_point = 0; q_point < n_face_q_points; ++q_point){
                    // 假设我们要施加一个沿 Y 轴向下的表面压力： [0, -100, 0]
                    Tensor<1, dim> traction;
                    traction[0] = 0.0;
                    traction[1] = -100.0;
                    traction[2] = 0.0;

                    for(const unsigned int i : fe_face_values.dof_indices()){
                        // 获取当前测试函数对应的物理分量 (0=X, 1=Y, 2=Z)
                        unsigned int component_i = fe.system_to_component_index(i).first;

                        // 将面力沿着该分量的部分积分，加到 cell_rhs 中
                        cell_rhs(i) += (
                            fe_face_values.shape_value(i, q_point) *
                            traction[component_i] *
                            fe_face_values.JxW(q_point)
                        );

                    }
                }
            }
        }

        // 将单元矩阵组装到全局
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs
        );
    }

    // ==========================================
    // 【核心修改】所有进程在循环结束后碰头，通过网络把边界缝合处的数据相加
    // ==========================================
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}

// template <int dim>
// void SoildBeam<dim>::solve(){
//     TimerOutput::Scope t(computing_timer, "3. solve");

//     SolverControl solver_control(100000, 1e-6 * system_rhs.l2_norm());
//     // SolverCG<Vector<double>> cg(solver_control);
//     SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);

//     // PreconditionSSOR<SparseMatrix<double>> preconditioner;
//     // preconditioner.initialize(system_matrix, 1.2);

//     // 呼叫 Trilinos 的 ILU (Block-Jacobi ILU)
//     TrilinosWrappers::PreconditionILU preconditioner;
//     preconditioner.initialize(system_matrix);
    

//     // cg.solve(system_matrix, solution, system_rhs, preconditioner);
//     cg.solve(system_matrix, solution, system_rhs, preconditioner);

//     // ==========================================
//     // 【新增】输出 CG 求解器的实际迭代步数
//     // ==========================================
//     // 只有 0 号核心负责打印迭代步数，避免被 4 个核心刷屏
//     if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0){
//         deallog << "   CG iterations: " << solver_control.last_step() << std::endl;
//     }

//     // 左边的 Vector<double> 是 deal.II 原生的单机串行向量。
//     // 右边的 solution 是 PETSc/Trilinos 的分布式向量。
//     // 执行过程: 所有节点同时向网络上广播自己手里那的解向量数据
//     // localized_solution是完整自由度的解向量
//     Vector<double> localized_solution(solution);
//     constraints.distribute(localized_solution);

//     solution = localized_solution;
// }
template <int dim>
void SoildBeam<dim>::solve(){
    TimerOutput::Scope t(computing_timer, "3. solve");

    nvtx3::scoped_range marker{"Solve_Phase"};

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

    SolverControl solver_control(dof_handler.n_dofs(), 1e-6 * system_rhs.l2_norm());
    LA::SolverCG solver(solver_control);

//     LA::MPI::PreconditionAMG::AdditionalData data;
//   #ifdef USE_PETSC_LA
//       data.symmetric_operator = true;
//   #else
//       /* Trilinos defaults are good */
//   #endif

    LA::MPI::PreconditionILU preconditioner;
    preconditioner.initialize(system_matrix);

    solver.solve(
        system_matrix,
        completely_distributed_solution,
        system_rhs,
        preconditioner
    );

    if(this_mpi_process == 0){
        deallog << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;
    }
    // 1 修正位移边界
    // 2 CG 求解器在求解时，矩阵里是没有悬挂节点的
    //      所以刚求出来的 completely_distributed_solution 里，主节点都有了正确的位移，但悬挂节点的值是错的（或者是 0）
    //      所以也是从新填悬挂节点的值
    constraints.distribute(completely_distributed_solution);
    // 触发了巨量的 MPI_Allgatherv 通信
    // locally_relevant_solution包括了幽灵节点的解
    // 虽然两个变量都是LA::MPI::Vector，但是初始化不同，内存等不同
    // 重载的=方法内部实现了mpi通信
    locally_relevant_solution = completely_distributed_solution;
}

// template <int dim>
// void SoildBeam<dim>::refine_grid(){
//     TimerOutput::Scope t(computing_timer, "4. refine_grid");

//     /*
//     家共享着同一张完整地图，那么每一次更新地图，所有人的动作必须做到 100% 绝对一致！
//     如果在细化网格时，0 号进程觉得 A 单元误差大，把它细分了；而 1 号进程觉得 A 单元误差小，没细分。
//     那么在下一秒进入 setup_system 时，这两个进程的网格拓扑就彻底对不上了，程序会瞬间崩溃。

//     */
//     // 第一步：强行看全图（获取完整位移场）
//     // 【极其关键】阶段一：强制把分布式的 solution 收集到单机的原生 Vector 里
//     // 这样所有的节点都有完整的位移场，可以各自完整地计算误差并细化全局网格
//     Vector<double> localized_solution(solution);

//     // 第二步：各扫门前雪（只算本地误差）
//     Vector<float> local_error_per_cell(triangulation.n_active_cells());
    
//     // KellyErrorEstimator<dim>::estimate(
//     //     dof_handler,
//     //     QGauss<dim-1>(fe.degree+1),
//     //     {},
//     //     solution,
//     //     estimated_error_per_cell
//     // );
//     KellyErrorEstimator<dim>::estimate(
//         dof_handler,
//         QGauss<dim - 1>(fe.degree + 1),
//         {},
//         localized_solution,
//         local_error_per_cell,
//         ComponentMask(),
//         nullptr,
//         MultithreadInfo::n_threads(),
//         this_mpi_process
//     );

//     // 第三步：【神级替换】丢掉恶心的 Trilinos 代数向量，直接用纯正的 MPI 底层通信！
//     // 意思是：把所有进程的 local_error_per_cell 加起来(MPI_SUM)，存到 localized_all_errors 里
//     Vector<float> localized_all_errors(triangulation.n_active_cells());
//     MPI_Allreduce(local_error_per_cell.begin(), // 1. 发送方（Send Buffer）：本地的残缺误差数组首地址
//                   localized_all_errors.begin(), // 2. 接收方（Recv Buffer）：用于存放最终结果的空数组首地址
//                   triangulation.n_active_cells(),   // 3. 数组长度：全图单元总数（比如 100,000）
//                   MPI_FLOAT,    // 4. 数据类型：因为 Vector<float> 里面存的是 float (32位浮点)
//                   MPI_SUM,      // 5. 归约操作（Operation）：相加！
//                   mpi_communicator);    // 6. 通信域：那个包含了 4 个克隆人的群聊

//     // 第四步：所有人拿着完整且一模一样的误差名单，同时切分网格
//     GridRefinement::refine_and_coarsen_fixed_number(
//         triangulation,
//         localized_all_errors, // 【修复 A】传入正确的变量名
//         0.1,
//         0.03
//     );

//     triangulation.execute_coarsening_and_refinement();
// }

template <int dim>
void SoildBeam<dim>::refine_grid()
{
    TimerOutput::Scope t(computing_timer, "4. refine_grid");

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
        dof_handler,
        QGauss<dim - 1>(fe.degree + 1),
        std::map<types::boundary_id, const Function<dim> *>(),
        locally_relevant_solution,
        estimated_error_per_cell
    );
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.1, 0.03
    );
    triangulation.execute_coarsening_and_refinement();
}

// template <int dim>
// void SoildBeam<dim>::output_results(const unsigned int cycle) const{
//     TimerOutput::Scope t(computing_timer, "5. output_results");

//     // 同理，收集到单机向量中，并强制只有 0 号节点负责写入文件！
//     // 否则 4 个节点同时往一个 vtk 里写，文件直接损坏。
//     // Vector<double> localized_solution(solution);

//     if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0){
//         DataOut<dim> data_out;
//         data_out.attach_dof_handler(dof_handler);

//         std::vector<std::string> solution_names;
//         switch (dim){
//             case 1:
//                 solution_names.emplace_back("displacement");
//                 break;
//             case 2:
//                 solution_names.emplace_back("x_displacement");
//                 solution_names.emplace_back("y_displacement");
//                 break;
//             case 3:
//                 solution_names.emplace_back("x_displacement");
//                 solution_names.emplace_back("y_displacement");
//                 solution_names.emplace_back("z_displacement");
//                 break;
//             default:
//                 DEAL_II_NOT_IMPLEMENTED();
//         }

//         data_out.add_data_vector(solution, solution_names);
//         data_out.build_patches();

//         std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
//         data_out.write_vtk(output);
//     }
// }

template <int dim>
void SoildBeam<dim>::output_results(const unsigned int cycle) const
{
    TimerOutput::Scope t(computing_timer, "output");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;
    solution_names.emplace_back("x_displacement");
    solution_names.emplace_back("y_displacement");
    solution_names.emplace_back("z_displacement");
    // 只能传入locally_relevant_solution
    
    data_out.add_data_vector(locally_relevant_solution, solution_names);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i){
        subdomain(i) = triangulation.locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
        "./", "solution", cycle, mpi_communicator, 2, 8);
}

template <int dim>
void SoildBeam<dim>::run(){

    deallog << "Running with "
#ifdef USE_PETSC_LA
    << "PETSc"
#else
    << "Trilinos"
#endif
    << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
    << " MPI rank(s)..." << std::endl;
    // 1. 在外面造一个空壳，保证它的生命周期足够长（比如一直活到 main 函数结束）
    // 此时它没有连接任何物理文件，绝对安全。
    std::ofstream log_file;

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0){
        // 1. 创建文件流
        log_file.open("log.txt");
        // 2. 将文件流挂载到 deal.II 的全局日志流上
        deallog.attach(log_file);
    }

    // (可选) 设置输出前缀的深度，让排版更整洁
    deallog.depth_console(2);

    for(unsigned int cycle=0; cycle < 10; ++cycle){
        // 在每轮循环开始时启动秒表
        Timer cycle_timer;

        if (this_mpi_process == 0) {
            // 现在只需要写一次！它会自动同时显示在屏幕终端并写入 log.txt
            deallog << "Cycle" << ": " << cycle << std::endl;
        }

        if(cycle==0){
            // 读取inp文件

            // 1. 创建网格导入工具对象
            GridIn<dim> grid_in;
            // 2. 将你的 triangulation (三角剖分/网格对象) 挂载到导入工具上
            grid_in.attach_triangulation(triangulation);
            // 3. 打开 inp 文件
            std::ifstream input_file(inp_path);
            // 4. 严谨的错误处理：如果文件路径不对，直接让程序报错并提示
            AssertThrow(input_file, ExcMessage("错误：无法打开网格文件 " + inp_path));
            // 5. 调用专门读取 Abaqus 格式的解析函数
            grid_in.read_abaqus(input_file);

            if (this_mpi_process == 0) {
                deallog << "Successfully read mesh from: " << inp_path << std::endl;
            }

            // 6. （极其关键的一步）网格读取完毕后，立刻调用你之前写的打标签函数！
            setup_boundary_ids();

        }else{
            refine_grid();
        }

        // if (this_mpi_process == 0) {
        //     deallog << "   Number of active cells:       "
        //                 << triangulation.n_active_cells() << std::endl;
        // }

        setup_system();
        // if (this_mpi_process == 0) {
        //     deallog << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        //                 << std::endl;
        // }

        assemble_system();
        solve();
        output_results(cycle);

        // 这是记录总时间
        // 在一轮循环结束时停止秒表，并输出时间
        cycle_timer.stop();
        if (this_mpi_process == 0) {
            deallog << "   Time taken for this cycle:    " 
                    << cycle_timer.wall_time() << " seconds." << std::endl;
            deallog << "-----------------------------------------------" << std::endl;
        }
        // 这是记录每部分的时间，有两个不同的文件
        // =========================================================
        // 【新增】在每一轮结束时，向文件写入华丽的分割线和当前 Cycle 标记
        // =========================================================
        if (this_mpi_process == 0) {
            timer_file << "\n=========================================================\n"
                    << "               Performance Summary: Cycle " << cycle << "\n"
                    << "               Active Cells: " << triangulation.n_active_cells() << "\n"
                    << "               DoFs:         " << dof_handler.n_dofs() << "\n"
                    << "=========================================================\n"
                    << std::flush; // 【新增】写完立刻强制推入硬盘！;
        }
        // 打印本轮的成绩单到 timer_file 里
        computing_timer.print_summary();
        
        // 【极其关键】一键清零！把总管小本子上的记录全部擦除，准备记录下一轮
        computing_timer.reset();
    }

    // 循环结束后，不再需要记录时，可以卸载文件
    deallog.detach();
}

int main(int argc, char *argv[]){
    // 【核心修改】初始化底层的 MPI 通信域、Trilinos 和相关环境
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // 【核心控制代码】限制每个 MPI 进程只能使用 2 个 TBB 线程
    dealii::MultithreadInfo::set_thread_limit(2);

    SoildBeam<3> soild_beam("../Job-1.inp");
    soild_beam.run();
}