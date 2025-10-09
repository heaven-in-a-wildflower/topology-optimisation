/*
To compile:
mpicxx -O3 -std=c++17 -fopenmp -I../mfem-4.8/ -I../hypre/src/hypre/include simp_transient.cpp -o simp_transient -L../mfem-4.8 -lmfem -L../hypre/src/hypre/lib -lHYPRE -L../metis-4.0 -lmetis -lrt -lopenblas

To run:
./simp_transient

To visualise:
./../glvis-4.4/glvis -m simp_transient_solns/transient_thermal.mesh -g simp_transient_solns/transient_thermal_final.gf
*/
//========================================================================
// Transient thermal topology optimization with time-dependent heating
#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

using namespace std;
using namespace mfem;
namespace fs = std::filesystem;

// ============================================================================
// COEFFICIENT CLASSES
// ============================================================================

// Time-dependent heat source (can be turned on/off)
class TransientHeatSourceCoeff : public Coefficient
{
private:
    double current_time;
    double heating_duration;
    double source_magnitude;

public:
    TransientHeatSourceCoeff(double heat_dur, double magnitude = 1.0)
        : current_time(0.0), heating_duration(heat_dur), source_magnitude(magnitude) {}

    void SetTime(double t) { current_time = t; }

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        // Only apply heat source during heating period
        if (current_time >= heating_duration)
            return 0.0;

        double coords[3];
        Vector x(coords, T.GetSpaceDim());
        T.Transform(ip, x);

        // Heat source along right edge
        if (x(0) < 0.01)
        //     return source_magnitude;
        // if (x(0) > 0.495 && x(0) < 0.505 && x(1) > 0.495 && x(1) < 0.505)
        // if ((x(0) > 0 && x(0) < 0.01 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.99 && x(0) < 1 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.99 && x(0) < 1 && x(1) > 0.99 && x(1) < 1) ||
        //     (x(0) > 0 && x(0) < 0.01 && x(1) > 0.99 && x(1) < 1))
            return source_magnitude;
        else
            return 0.0;
    }
};

// Convection coefficient
class LocalConvectionCoeff : public Coefficient
{
public:
    double hval;
    LocalConvectionCoeff(double h) : hval(h) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        Vector x;
        T.Transform(ip, x);

        // Convection on left edge
        // if ((x(0) > 0 && x(0) < 0.01 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.99 && x(0) < 1 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.99 && x(0) < 1 && x(1) > 0.99 && x(1) < 1) ||
        //     (x(0) > 0 && x(0) < 0.01 && x(1) > 0.99 && x(1) < 1))
        // if (x(0)<0.01 && x(1)>0.45 && x(1)<0.55)
        if(x(0)>0.99)
            return hval;
        else
            return 0.0;
    }
};

// h * T∞ coefficient for convection BC
class LocalConvectionAmbientCoeff : public Coefficient
{
public:
    double hval, Tamb;
    LocalConvectionAmbientCoeff(double h, double T) : hval(h), Tamb(T) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        Vector x;
        T.Transform(ip, x);

        // if (x(0) < 0.01)
        // if ((x(0) > 0 && x(0) < 0.01 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.99 && x(0) < 1 && x(1) > 0 && x(1) < 0.01) ||
        //     (x(0) > 0.99 && x(0) < 1 && x(1) > 0.99 && x(1) < 1) ||
        //     (x(0) > 0 && x(0) < 0.01 && x(1) > 0.99 && x(1) < 1))
        // if (x(0)<0.01 && x(1)>0.45 && x(1)<0.55)
        if(x(0)>0.99)
            return hval * Tamb;
        else
            return 0.0;
    }
};

// SIMP thermal conductivity
class ThermalConductivityCoeff : public Coefficient
{
private:
    GridFunction *rho;
    double simp_exponent;
    double rho_min;

public:
    ThermalConductivityCoeff(GridFunction *rho_, double simp_exp, double rho_min_val)
        : rho(rho_), simp_exponent(simp_exp), rho_min(rho_min_val) {}

    void SetSIMPExponent(double exp) { simp_exponent = exp; }

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        double density = rho->GetValue(T, ip);
        density = max(rho_min, min(1.0, density));

        // k = k_min + (k_max - k_min) * ρ^p
        double k_min = rho_min * 1e-3;
        double k_max = 1.0;

        return k_min + (k_max - k_min) * pow(density, simp_exponent);
    }
};

// Thermal capacity coefficient: ρ_material * c_p * ρ_design
class ThermalCapacityCoeff : public Coefficient
{
private:
    GridFunction *rho;
    double rho_material;
    double cp;
    double rho_min;

public:
    ThermalCapacityCoeff(GridFunction *rho_, double rho_mat, double cp_val, double rho_min_val)
        : rho(rho_), rho_material(rho_mat), cp(cp_val), rho_min(rho_min_val) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        double density = rho->GetValue(T, ip);
        density = max(rho_min, min(1.0, density));

        // Thermal capacity
        return rho_material * cp * density;
    }
};

// ============================================================================
// DENSITY FILTER
// ============================================================================

void densityFilterPDE(Mesh &mesh, GridFunction &rho, GridFunction &rho_filtered,
                      double r_min, Array<int> *ess_bdr = nullptr)
{
    FiniteElementSpace *fes = rho.FESpace();

    BilinearForm filter_form(fes);
    ConstantCoefficient r_squared(r_min * r_min);
    ConstantCoefficient one(1.0);

    filter_form.AddDomainIntegrator(new DiffusionIntegrator(r_squared));
    filter_form.AddDomainIntegrator(new MassIntegrator(one));
    filter_form.Assemble();
    filter_form.Finalize();

    BilinearForm mass_form(fes);
    mass_form.AddDomainIntegrator(new MassIntegrator(one));
    mass_form.Assemble();
    mass_form.Finalize();

    Vector rhs(fes->GetNDofs());
    mass_form.Mult(rho, rhs);

    Array<int> empty_ess_tdof;
    if (ess_bdr == nullptr)
    {
        ess_bdr = &empty_ess_tdof;
    }

    SparseMatrix &A = filter_form.SpMat();

    CGSolver cg;
    cg.SetMaxIter(1000);
    cg.SetRelTol(1e-12);
    cg.SetAbsTol(1e-15);
    cg.SetPrintLevel(0);
    cg.SetOperator(A);

    rho_filtered = 0.0;
    cg.Mult(rhs, rho_filtered);

    if (!cg.GetConverged())
    {
        std::cerr << "Warning: Filter did not converge!" << std::endl;
    }
}

// ============================================================================
// TRANSIENT SENSITIVITY CALCULATION (Simplified Direct Method)
// ============================================================================

void ComputeTransientSensitivities(
    const vector<GridFunction> &temperature_history,
    const GridFunction &rho,
    GridFunction &sensitivity,
    double simp_exponent,
    double rho_min,
    double dt,
    FiniteElementSpace *fes)
{
    Mesh *mesh = fes->GetMesh();
    int ne = mesh->GetNE();
    sensitivity = 0.0;

    Vector elem_sens, elem_rho;
    DenseMatrix dshape;
    Vector grad_T;

    // Loop over all time steps
    for (size_t t_idx = 0; t_idx < temperature_history.size(); t_idx++)
    {
        const GridFunction &temperature = temperature_history[t_idx];

        // Loop over elements
        for (int e = 0; e < ne; e++)
        {
            const FiniteElement *fe = fes->GetFE(e);
            ElementTransformation *T = mesh->GetElementTransformation(e);
            int dof = fe->GetDof();
            int dim = fe->GetDim();

            Array<int> vdofs;
            fes->GetElementVDofs(e, vdofs);

            elem_rho.SetSize(dof);
            Vector elem_temp(dof);
            elem_sens.SetSize(dof);
            rho.GetSubVector(vdofs, elem_rho);
            temperature.GetSubVector(vdofs, elem_temp);
            elem_sens = 0.0;

            const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder());
            dshape.SetSize(dof, dim);
            grad_T.SetSize(dim);

            for (int i = 0; i < ir->GetNPoints(); i++)
            {
                const IntegrationPoint &ip = ir->IntPoint(i);
                T->SetIntPoint(&ip);
                double w = T->Weight() * ip.weight;

                Vector shape(dof);
                fe->CalcShape(ip, shape);
                fe->CalcDShape(ip, dshape);

                double rho_ip = 0.0;
                for (int j = 0; j < dof; j++)
                    rho_ip += elem_rho(j) * shape(j);
                rho_ip = max(rho_min, min(1.0, rho_ip));

                Vector grad_T_ref(dim);
                grad_T_ref = 0.0;
                for (int j = 0; j < dof; j++)
                    for (int d = 0; d < dim; d++)
                        grad_T_ref(d) += elem_temp(j) * dshape(j, d);

                T->InverseJacobian().MultTranspose(grad_T_ref, grad_T);

                double grad_T_squared = 0.0;
                for (int d = 0; d < dim; d++)
                    grad_T_squared += grad_T(d) * grad_T(d);

                // Sensitivity: dk/drho * |∇T|²
                double k_min = rho_min * 1e-3;
                double k_max = 1.0;
                double dk_drho = (k_max - k_min) * simp_exponent * pow(rho_ip, simp_exponent - 1.0);
                double local_sensitivity = -dk_drho * grad_T_squared * dt;

                for (int j = 0; j < dof; j++)
                    elem_sens(j) += local_sensitivity * shape(j) * w;
            }

            sensitivity.AddElementVector(vdofs, elem_sens);
        }
    }
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    // ========================================================================
    // PARAMETERS
    // ========================================================================

    // Material properties
    const double rho_material = 2700.0; // kg/m³ (aluminum)
    const double cp = 900.0;            // J/(kg·K) (specific heat)
    const double k_thermal = 100.0;     // W/(m·K) (normalized)

    // Time parameters
    const double heating_duration = 10.0; // seconds (KEY PARAMETER TO VARY!)
    const double t_final = 20.0;          // Total simulation time
    const double dt = 0.05;               // Time step
    const int num_time_steps = (int)(t_final / dt);

    // Optimization parameters
    const double vol_frac = 0.3;
    const double rho_min = 1e-6;
    const int max_iter = 150;
    const int non_penalized_iter = 10;
    const double max_simp_exp = 8.0;
    const double tolerance = 1e-5;
    const double move_limit = 0.05;
    const double r_min = 0.02;

    // Boundary conditions
    const double h_val = 10.0; // Convection coefficient
    const double T_amb = 25.0; // Ambient temperature

    cout << "==================================================" << endl;
    cout << "TRANSIENT THERMAL TOPOLOGY OPTIMIZATION" << endl;
    cout << "==================================================" << endl;
    cout << "Heating duration: " << heating_duration << " seconds" << endl;
    cout << "Total simulation time: " << t_final << " seconds" << endl;
    cout << "Time step: " << dt << " seconds" << endl;
    cout << "Number of time steps: " << num_time_steps << endl;
    cout << "Target volume fraction: " << vol_frac << endl;
    cout << "==================================================" << endl;

    // ========================================================================
    // MESH AND FINITE ELEMENT SPACE
    // ========================================================================

    Mesh mesh = Mesh::MakeCartesian2D(80, 80, Element::QUADRILATERAL,
                                      false, 1.0, 1.0, false);

    H1_FECollection fec(1, mesh.Dimension());
    FiniteElementSpace fes(&mesh, &fec);

    cout << "Number of unknowns: " << fes.GetNDofs() << endl;

    // ========================================================================
    // BOUNDARY CONDITIONS
    // ========================================================================

    Array<int> ess_tdof_cold;
    double tol = 1e-6;

    for (int i = 0; i < fes.GetNBE(); i++)
    {
        Array<int> vdofs;
        fes.GetBdrElementVDofs(i, vdofs);

        for (int j = 0; j < vdofs.Size(); j++)
        {
            int dof = vdofs[j];
            if (dof < 0)
                dof = -1 - dof;

            int vertex_id = dof;
            const double *vertex = mesh.GetVertex(vertex_id);
            double x = vertex[0];

            if (x < 0.01 + tol)
            {
                if (ess_tdof_cold.Find(dof) < 0)
                    ess_tdof_cold.Append(dof);
            }
        }
    }

    cout << "Cold boundary DOFs: " << ess_tdof_cold.Size() << endl;

    // ========================================================================
    // INITIALIZE FIELDS
    // ========================================================================

    GridFunction rho(&fes), rho_filtered(&fes), sensitivity(&fes);
    rho = vol_frac;
    sensitivity = 0.0;

    // ========================================================================
    // SETUP COEFFICIENTS
    // ========================================================================

    TransientHeatSourceCoeff source_coeff(heating_duration, 10.0);
    LocalConvectionCoeff h_coeff(h_val);
    LocalConvectionAmbientCoeff hTinf_coeff(h_val, T_amb);

    // ========================================================================
    // OPTIMIZATION LOOP
    // ========================================================================

    double simp_exponent = 1.0;
    int exp_update_counter = 0;
    vector<double> compliance_history;
    double old_compliance = 1e30;

    CGSolver cg_solver;
    cg_solver.SetMaxIter(5000);
    cg_solver.SetRelTol(1e-8);
    cg_solver.SetAbsTol(1e-12);
    cg_solver.SetPrintLevel(0);

    for (int iter = 0; iter < max_iter; iter++)
    {
        cout << "\n================================================" << endl;
        cout << "Iteration " << iter + 1 << "/" << max_iter;
        cout << " (SIMP p=" << simp_exponent << ")" << endl;
        cout << "================================================" << endl;

        // ====================================================================
        // STEP 1: FILTER DESIGN VARIABLES
        // ====================================================================
        densityFilterPDE(mesh, rho, rho_filtered, r_min);

        // ====================================================================
        // STEP 2: TRANSIENT FORWARD ANALYSIS
        // ====================================================================

        ThermalConductivityCoeff k_coeff(&rho_filtered, simp_exponent, rho_min);
        ThermalCapacityCoeff rho_cp_coeff(&rho_filtered, rho_material, cp, rho_min);

        GridFunction temperature(&fes);
        temperature = T_amb;

        // Set cold boundary
        // for (int i = 0; i < ess_tdof_cold.Size(); i++)
        // {
        //     temperature[ess_tdof_cold[i]] = 0.0;
        // }

        vector<GridFunction> temperature_history;
        double compliance_transient = 0.0;

        cout << "Running transient analysis..." << endl;

        // Assemble mass matrix (thermal capacity)
        BilinearForm mass_form(&fes);
        mass_form.AddDomainIntegrator(new MassIntegrator(rho_cp_coeff));
        mass_form.Assemble();
        mass_form.Finalize();

        for (int t_step = 0; t_step < num_time_steps; t_step++)
        {
            double time = t_step * dt;
            source_coeff.SetTime(time);

            // Stiffness matrix (conductivity + convection)
            BilinearForm k_form(&fes);
            k_form.AddDomainIntegrator(new DiffusionIntegrator(k_coeff));
            k_form.AddBoundaryIntegrator(new BoundaryMassIntegrator(h_coeff));
            k_form.Assemble();
            k_form.Finalize();

            // Combined system: M/dt + K
            SparseMatrix &M = mass_form.SpMat();
            SparseMatrix &K = k_form.SpMat();
            SparseMatrix *A = Add(1.0 / dt, M, 1.0, K);

            // Right-hand side: M/dt * T_n + F(t)
            Vector rhs(fes.GetNDofs());
            M.Mult(temperature, rhs);
            rhs *= 1.0 / dt;

            LinearForm q_form(&fes);
            q_form.AddDomainIntegrator(new DomainLFIntegrator(source_coeff));
            q_form.AddBoundaryIntegrator(new BoundaryLFIntegrator(hTinf_coeff));
            q_form.Assemble();
            rhs += q_form;

            // Form linear system with BCs
            OperatorPtr A_bc;
            Vector B, X;
            Array<int> ess_tdof_list;
            // ess_tdof_list.Append(ess_tdof_cold);

            // Create temporary bilinear form for FormLinearSystem
            BilinearForm temp_form(&fes);
            temp_form.AddDomainIntegrator(new MassIntegrator());
            temp_form.Assemble();
            temp_form.Finalize();

            // Copy combined matrix into temp form
            temp_form.SpMat() = *A;

            LinearForm temp_rhs(&fes);
            temp_rhs = 0.0;

            temp_form.FormLinearSystem(ess_tdof_list, temperature, temp_rhs, A_bc, X, B);

            // Update RHS
            for (int i = 0; i < B.Size(); i++)
                B(i) = rhs(i);

            // Solve
            cg_solver.SetOperator(*A_bc);
            cg_solver.Mult(B, X);

            // Recover solution
            temp_form.RecoverFEMSolution(X, temp_rhs, temperature);

            // Store temperature for sensitivity analysis
            GridFunction temp_copy(&fes);
            temp_copy = temperature;
            temperature_history.push_back(temp_copy);

            // Accumulate compliance
            compliance_transient += (q_form * temperature) * dt;

            delete A;

            if (t_step % 50 == 0)
                cout << "  Time step " << t_step << "/" << num_time_steps
                     << " (t=" << time << "s)" << endl;
        }

        compliance_history.push_back(compliance_transient);

        // Volume calculation
        double current_vol = 0.0;
        for (int i = 0; i < rho_filtered.Size(); i++)
            current_vol += rho_filtered(i);
        current_vol /= rho_filtered.Size();

        cout << "\nTransient compliance: " << compliance_transient << endl;
        cout << "Volume fraction: " << current_vol << endl;

        // ====================================================================
        // STEP 3: SENSITIVITY ANALYSIS
        // ====================================================================

        GridFunction sensitivity_physical(&fes);
        ComputeTransientSensitivities(temperature_history, rho_filtered,
                                      sensitivity_physical, simp_exponent,
                                      rho_min, dt, &fes);

        // Chain rule: filter adjoint
        densityFilterPDE(mesh, sensitivity_physical, sensitivity, r_min);

        double min_sens = 1e30, max_sens = -1e30;
        for (int i = 0; i < sensitivity.Size(); i++)
        {
            min_sens = min(min_sens, sensitivity(i));
            max_sens = max(max_sens, sensitivity(i));
        }
        cout << "Sensitivity range: [" << min_sens << ", " << max_sens << "]" << endl;

        // ====================================================================
        // STEP 4: OPTIMALITY CRITERIA UPDATE
        // ====================================================================

        Vector rho_old = rho;
        double lam_min = 1e-10, lam_max = 1e10;

        for (int lam_iter = 0; lam_iter < 50; lam_iter++)
        {
            double lam_mid = 0.5 * (lam_min + lam_max);
            double vol_sum = 0.0;

            for (int i = 0; i < rho.Size(); i++)
            {
                double rho_old_i = rho_old(i);
                double sens_i = sensitivity(i);
                double Be = -sens_i / lam_mid;

                double rho_new_i;
                if (Be <= 0)
                {
                    rho_new_i = max(rho_min, rho_old_i - move_limit);
                }
                else
                {
                    rho_new_i = max(rho_min, min(1.0, rho_old_i * sqrt(Be)));
                    rho_new_i = max(rho_old_i - move_limit, rho_new_i);
                    rho_new_i = min(rho_old_i + move_limit, rho_new_i);
                }

                rho_new_i = max(rho_min, min(1.0, rho_new_i));
                vol_sum += rho_new_i;
            }

            double vol_current = vol_sum / rho.Size();

            if (abs(vol_current - vol_frac) < 1e-6)
                break;

            if (vol_current > vol_frac)
                lam_min = lam_mid;
            else
                lam_max = lam_mid;
        }

        // Apply final update
        double lam_final = 0.5 * (lam_min + lam_max);
        double max_change = 0.0;

        for (int i = 0; i < rho.Size(); i++)
        {
            double rho_old_i = rho_old(i);
            double sens_i = sensitivity(i);
            double Be = -sens_i / lam_final;

            double rho_new_i;
            if (Be <= 0)
            {
                rho_new_i = max(rho_min, rho_old_i - move_limit);
            }
            else
            {
                rho_new_i = max(rho_min, min(1.0, rho_old_i * sqrt(Be)));
                rho_new_i = max(rho_old_i - move_limit, rho_new_i);
                rho_new_i = min(rho_old_i + move_limit, rho_new_i);
            }

            rho_new_i = max(rho_min, min(1.0, rho_new_i));
            max_change = max(max_change, abs(rho_new_i - rho_old_i));
            rho(i) = rho_new_i;
        }

        cout << "Max density change: " << max_change << endl;

        // ====================================================================
        // STEP 5: CONTINUATION
        // ====================================================================

        exp_update_counter++;

        if (iter >= non_penalized_iter)
        {
            if (iter == non_penalized_iter)
                cout << "\n*** Starting penalization ***" << endl;

            if (exp_update_counter >= 5 && simp_exponent < max_simp_exp)
            {
                simp_exponent = min(simp_exponent * 1.5, max_simp_exp);
                exp_update_counter = 0;
                cout << "⚡ SIMP exponent increased to: " << simp_exponent << endl;
            }
        }

        // ====================================================================
        // STEP 6: CONVERGENCE CHECK
        // ====================================================================

        int binary_count = 0;
        for (int i = 0; i < rho_filtered.Size(); i++)
        {
            double rho_val = rho_filtered(i);
            if (rho_val < 0.05 || rho_val > 0.95)
                binary_count++;
        }
        double binary_percent = (double)binary_count / rho_filtered.Size() * 100.0;
        cout << "Binary design: " << binary_percent << "%" << endl;

        if (iter > 15 && abs(compliance_transient - old_compliance) < tolerance * abs(old_compliance))
        {
            cout << "\n✓ Converged after " << iter + 1 << " iterations!" << endl;
            break;
        }

        old_compliance = compliance_transient;

        // ====================================================================
        // STEP 7: OUTPUT
        // ====================================================================

        // if ((iter + 1) % 25 == 0 || iter == 0)
        // {
        //     string filename = "transient_thermal_" + to_string(iter + 1) + ".gf";
        //     ofstream sol_ofs(filename.c_str());
        //     rho_filtered.Save(sol_ofs);
        // }
    }

    // ========================================================================
    // SAVE FINAL RESULTS
    // ========================================================================

    string result_dir = "simp_transient_solns/";
    fs::create_directories(result_dir);

    ofstream mesh_ofs(result_dir + "transient_thermal.mesh");
    mesh.Print(mesh_ofs);

    ofstream final_rho(result_dir + "transient_thermal_final.gf");
    rho_filtered.Save(final_rho);

    GridFunction rho_binary(&fes);
    for (int i = 0; i < rho.Size(); i++)
        rho_binary(i) = (rho_filtered(i) > 0.5) ? 1.0 : rho_min;

    ofstream binary_rho(result_dir + "transient_thermal_binary.gf");
    rho_binary.Save(binary_rho);

    cout << "\n==================================================" << endl;
    cout << "OPTIMIZATION COMPLETE" << endl;
    cout << "==================================================" << endl;
    cout << "Compliance improvement: "
         << (compliance_history[0] - compliance_history.back()) / compliance_history[0] * 100.0
         << "%" << endl;
    cout << "Final SIMP exponent: " << simp_exponent << endl;
    cout << "Heating duration used: " << heating_duration << " seconds" << endl;
    cout << "\nVisualize with:" << endl;
    cout << "glvis -m transient_results/transient_thermal.mesh -g transient_results/transient_thermal_final.gf" << endl;

    MPI_Finalize();
    return 0;
}