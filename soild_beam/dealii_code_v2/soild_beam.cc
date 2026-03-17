#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/timer.h>

// --- 【新增】MPI 与并行网格分割工具 ---
#include <deal.II/base/mpi.h>
#include <deal.II/base/index_set.h>
#include <deal.II/grid/grid_tools.h>

// --- 【新增】Trilinos 并行代数库 ---
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/sparsity_tools.h>

#include <fstream>
#include <iostream>
#include <string>

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

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;

    const FESystem<dim> fe;

    AffineConstraints<double> constraints;

    // 【新增】MPI 通信器与自由度归属账本
    MPI_Comm mpi_communicator;
    IndexSet locally_owned_dofs;    // 属于我的自由度
    IndexSet locally_relevant_dofs; // 我能看见的自由度（含幽灵节点）

    // 【修改】原生的 SparsityPattern 和 SparseMatrix 替换为 Trilinos
    TrilinosWrappers::SparseMatrix system_matrix;

    // 串行版本代码
    // SparsityPattern sparsity_pattern;
    // SparseMatrix<double> system_matrix;

    // 【修改】原生 Vector 替换为 Trilinos::MPI::Vector
    TrilinosWrappers::MPI::Vector solution;
    TrilinosWrappers::MPI::Vector system_rhs;

    // Vector<double> solution;
    // Vector<double> system_rhs;

    // inp文件路径
    string inp_path;

    // 1. 先声明文件流（必须在计时器前面！）
    std::ofstream timer_file;
    // 【新增】高级性能剖析计时器
    // output_results声明为const
    // 在output_results内是需要修改TimerOutput的，添加mutable
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
dof_handler(triangulation), fe(FE_Q<dim>(1) ^ dim), inp_path(inp_path),
timer_file("timer_summary.txt"),
computing_timer(mpi_communicator, timer_file, TimerOutput::summary, TimerOutput::wall_times){

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

// 重写函数
// template<int dim>
// void SoildBeam<dim>::setup_system(){
//     // 【新增】哨兵：它会自动记录整个 setup_system 函数的耗时，标签名为 "1. Assemble system"
//     TimerOutput::Scope t(computing_timer, "1. setup_system");

//     dof_handler.distribute_dofs(fe);
//     solution.reinit(dof_handler.n_dofs());
//     system_rhs.reinit(dof_handler.n_dofs());

//     constraints.clear();
//     DoFTools::make_hanging_node_constraints(dof_handler, constraints);
//     VectorTools::interpolate_boundary_values(
//         dof_handler,
//         types::boundary_id(1),
//         Functions::ZeroFunction<dim>(dim),
//         constraints
//     );
//     constraints.close();

//     DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
//     DoFTools::make_sparsity_pattern(
//         dof_handler,
//         dsp,
//         constraints,
//         /**keep_constrained_dofs = */false
//     );
//     sparsity_pattern.copy_from(dsp);

//     system_matrix.reinit(sparsity_pattern);
// }
template <int dim>
void SoildBeam<dim>::setup_system(){
    TimerOutput::Scope t(computing_timer, "1. setup_system");

    // 获取当前 MPI 环境的总核数和自己的编号
    const unsigned int n_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_communicator);

    // 【核心 1】呼叫 METIS 将全局网格打上分区标签
    GridTools::partition_triangulation(n_mpi_processes, triangulation);

    dof_handler.distribute_dofs(fe);

    // 【核心 2】获取自由度账本
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // 初始化带幽灵节点的分布式向量
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    // 约束必须基于 relevant_dofs 构建，否则处理悬挂节点会崩溃
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
        dof_handler, types::boundary_id(1), Functions::ZeroFunction(dim), constraints
    );
    constraints.close();

    // 【核心 3】通过 MPI 自动构建和分发矩阵稀疏图
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DofTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp, locally_owned_dofs, mpi_communicator, locally_owned_dofs);

    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}


template<int dim>
void SoildBeam<dim>::assemble_system(){
    // 【新增】哨兵：它会自动记录整个 assemble_system 函数的耗时，标签名为 "2. Assemble system"
    TimerOutput::Scope t(computing_timer, "2. assemble_system");
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

    // 【新增】获取当前 MPI 进程的编号
    const unsigned int this_mpi_process = Utilities::MPI::this_mpi_process(mpi_communicator);

    // 对所有单元进行遍历
    for(const auto &cell : dof_handler.active_cell_iterators()){
        // ==========================================
        // 【核心修改】只处理属于当前进程的单元！
        // ==========================================
        if(cell->subdomain_id() != this_mpi_process){
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

template <int dim>
void SoildBeam<dim>::solve(){
    TimerOutput::Scope t(computing_timer, "3. solve");

    SolverControl solver_control(100000, 1e-6 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    // ==========================================
    // 【新增】输出 CG 求解器的实际迭代步数
    // ==========================================
    deallog << "   CG iterations: " << solver_control.last_step() << std::endl;

    constraints.distribute(solution);
}

template <int dim>
void SoildBeam<dim>::refine_grid(){
    TimerOutput::Scope t(computing_timer, "4. refine_grid");

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    
    KellyErrorEstimator<dim>::estimate(
        dof_handler,
        QGauss<dim-1>(fe.degree+1),
        {},
        solution,
        estimated_error_per_cell
    );

    GridRefinement::refine_and_coarsen_fixed_number(
        triangulation,
        estimated_error_per_cell,
        0.1,
        0.03
    );

    triangulation.execute_coarsening_and_refinement();
}

template <int dim>
void SoildBeam<dim>::output_results(const unsigned int cycle) const{
    TimerOutput::Scope t(computing_timer, "5. output_results");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;
    switch (dim){
        case 1:
            solution_names.emplace_back("displacement");
            break;
        case 2:
            solution_names.emplace_back("x_displacement");
            solution_names.emplace_back("y_displacement");
            break;
        case 3:
            solution_names.emplace_back("x_displacement");
            solution_names.emplace_back("y_displacement");
            solution_names.emplace_back("z_displacement");
            break;
        default:
            DEAL_II_NOT_IMPLEMENTED();
    }

    data_out.add_data_vector(solution, solution_names);
    data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
}

template <int dim>
void SoildBeam<dim>::run(){
    // 1. 创建文件流
    ofstream log("log.txt");

    // 2. 将文件流挂载到 deal.II 的全局日志流上
    deallog.attach(log);

    // (可选) 设置输出前缀的深度，让排版更整洁
    deallog.depth_console(2);

    for(unsigned int cycle=0; cycle < 10; ++cycle){
        // 在每轮循环开始时启动秒表
        Timer cycle_timer;

        // 现在只需要写一次！它会自动同时显示在屏幕终端并写入 log.txt
        deallog << "Cycle" << ": " << cycle << std::endl;

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

            deallog << "Successfully read mesh from: " << inp_path << std::endl;

            // 6. （极其关键的一步）网格读取完毕后，立刻调用你之前写的打标签函数！
            setup_boundary_ids();

        }else{
            refine_grid();
        }

        deallog << "   Number of active cells:       "
                    << triangulation.n_active_cells() << std::endl;

        setup_system();
        deallog << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                    << std::endl;

        assemble_system();
        solve();
        output_results(cycle);

        // 这是记录总时间
        // 在一轮循环结束时停止秒表，并输出时间
        cycle_timer.stop();
        deallog << "   Time taken for this cycle:    " 
                << cycle_timer.wall_time() << " seconds." << std::endl;
        deallog << "-----------------------------------------------" << std::endl;
        // 这是记录每部分的时间，有两个不同的文件
        // =========================================================
        // 【新增】在每一轮结束时，向文件写入华丽的分割线和当前 Cycle 标记
        // =========================================================
        timer_file << "\n=========================================================\n"
                   << "               Performance Summary: Cycle " << cycle << "\n"
                   << "               Active Cells: " << triangulation.n_active_cells() << "\n"
                   << "               DoFs:         " << dof_handler.n_dofs() << "\n"
                   << "=========================================================\n";
                   
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

    SoildBeam<3> soild_beam("../Job-1.inp");
    soild_beam.run();
}