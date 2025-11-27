/*
To compile:
mpicxx -O3 -std=c++17 -fopenmp -I../mfem-4.8/ -I../hypre/src/hypre/include simp_3d.cpp -o simp_3d -L../mfem-4.8 -lmfem -L../hypre/src/hypre/lib -lHYPRE -L../metis-4.0 -lmetis -lrt -lopenblas 

To run:
./simp_3d

To visualise:
./../glvis-4.4/glvis -m simp_3d_solns/cylinder_thermal.mesh -g simp_3d_solns/cylinder_thermal_final.gf 
*/

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cmath>

using namespace std;
using namespace mfem;
namespace fs = std::filesystem;

// ============================================================================
// GLOBAL PARAMETERS FOR GEOMETRY (for coefficient access)
// ============================================================================
struct CylinderGeometry {
    double radius;
    double height;
    double center_x;
    double center_y;
    double center_z;  // z-axis aligned cylinder
    double fixed_zone_fraction;  // fraction of height at top/bottom to keep solid
    
    CylinderGeometry(double r = 0.5, double h = 2.0, double cx = 0.0, double cy = 0.0, double cz = 0.0, double fzf = 0.05)
        : radius(r), height(h), center_x(cx), center_y(cy), center_z(cz), fixed_zone_fraction(fzf) {}
    
    bool isOnTopFace(const Vector& x, double tol = 1e-3) const {
        double z = x(2);
        double top_z = center_z + height / 2.0;
        return abs(z - top_z) < tol;
    }
    
    bool isOnBottomFace(const Vector& x, double tol = 1e-3) const {
        double z = x(2);
        double bottom_z = center_z - height / 2.0;
        return abs(z - bottom_z) < tol;
    }
    
    bool isOnCylindricalSurface(const Vector& x, double tol = 1e-3) const {
        double dx = x(0) - center_x;
        double dy = x(1) - center_y;
        double r = sqrt(dx*dx + dy*dy);
        return abs(r - radius) < tol;
    }
    
    bool isInFixedZone(const Vector& x, double tol = 1e-6) const {
        double z = x(2);
        double top_z = center_z + height / 2.0;
        double bottom_z = center_z - height / 2.0;
        double fixed_height = height * fixed_zone_fraction;
        
        return (z >= (top_z - fixed_height - tol)) || (z <= (bottom_z + fixed_height + tol));
    }
};

// Global geometry object (accessible to coefficients)
CylinderGeometry GEOM;

// ============================================================================
// DEBUG LOGGER
// ============================================================================
class DebugLogger {
private:
    ofstream log_file;
    bool enabled;
    
public:
    DebugLogger(const string& filename, bool enable = true) : enabled(enable) {
        if (enabled) {
            log_file.open(filename);
            log_file << "=== DEBUG LOG ===" << endl;
            log_file << "Timestamp: " << __DATE__ << " " << __TIME__ << endl;
            log_file << "==================" << endl << endl;
        }
    }
    
    ~DebugLogger() {
        if (log_file.is_open()) log_file.close();
    }
    
    template<typename T>
    void log(const string& label, const T& value) {
        if (!enabled) return;
        log_file << "[DEBUG] " << label << ": " << value << endl;
        log_file.flush();
    }
    
    void section(const string& title) {
        if (!enabled) return;
        log_file << "\n======================================" << endl;
        log_file << "  " << title << endl;
        log_file << "======================================" << endl;
        log_file.flush();
    }
};

// Global debug logger
DebugLogger DEBUG_LOG("simp_3d_debug.log", true);

// ============================================================================
// COEFFICIENT CLASSES
// ============================================================================

// Heat flux on top face (constant)
class TopFaceHeatFluxCoeff : public Coefficient
{
private:
    double flux_magnitude;

public:
    TopFaceHeatFluxCoeff(double flux = 1000.0) : flux_magnitude(flux) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        Vector x(3);
        T.Transform(ip, x);
        
        if (GEOM.isOnTopFace(x)) {
            return flux_magnitude;
        }
        return 0.0;
    }
};

// Convection on cylindrical surface and bottom (not top)
class CylindricalConvectionCoeff : public Coefficient
{
private:
    double hval;
    
public:
    CylindricalConvectionCoeff(double h) : hval(h) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        Vector x(3);
        T.Transform(ip, x);
        
        // Convection on cylindrical surface and bottom face
        // NOT on top face (that's where heat flux is applied)
        if (GEOM.isOnBottomFace(x)) {
            return hval;
        }
        return 0.0;
    }
};

// h * T_amb for convection BC
class CylindricalConvectionAmbientCoeff : public Coefficient
{
private:
    double hval, Tamb;
    
public:
    CylindricalConvectionAmbientCoeff(double h, double T) : hval(h), Tamb(T) {}

    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
    {
        Vector x(3);
        T.Transform(ip, x);
        
        if (GEOM.isOnBottomFace(x)) {
            return hval * Tamb;
        }
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

        double k_min = rho_min * 1e-3;
        double k_max = 1.0;

        return k_min + (k_max - k_min) * pow(density, simp_exponent);
    }
};

// ============================================================================
// DENSITY FILTER (3D)
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
    if (ess_bdr == nullptr) {
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

    if (!cg.GetConverged()) {
        DEBUG_LOG.log("Filter convergence", "FAILED");
        cerr << "Warning: Filter did not converge!" << endl;
    } else {
        DEBUG_LOG.log("Filter convergence", "OK");
    }
}

// ============================================================================
// SENSITIVITY CALCULATION (3D)
// ============================================================================

void ComputeSteadyStateSensitivities(
    const GridFunction &temperature,
    const GridFunction &rho,
    GridFunction &sensitivity,
    double simp_exponent,
    double rho_min,
    FiniteElementSpace *fes)
{
    Mesh *mesh = fes->GetMesh();
    int ne = mesh->GetNE();
    sensitivity = 0.0;

    Vector elem_sens, elem_rho;
    DenseMatrix dshape;
    Vector grad_T;

    double total_sensitivity_magnitude = 0.0;
    int elem_count = 0;

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

            double k_min = rho_min * 1e-3;
            double k_max = 1.0;
            double dk_drho = (k_max - k_min) * simp_exponent * pow(rho_ip, simp_exponent - 1.0);

            double local_sensitivity = -dk_drho * grad_T_squared;

            for (int j = 0; j < dof; j++)
                elem_sens(j) += local_sensitivity * shape(j) * w;
        }

        sensitivity.AddElementVector(vdofs, elem_sens);
        
        for (int j = 0; j < elem_sens.Size(); j++)
            total_sensitivity_magnitude += abs(elem_sens(j));
        elem_count++;
    }

    DEBUG_LOG.log("Avg sensitivity magnitude", total_sensitivity_magnitude / elem_count);
}

// ============================================================================
// IDENTIFY FIXED DOFS
// ============================================================================

void IdentifyFixedDOFs(Mesh &mesh, const FiniteElementSpace &fes, Array<int> &fixed_dofs)
{
    fixed_dofs.DeleteAll();
    int ndofs = fes.GetNDofs();
    
    DEBUG_LOG.section("Fixed DOF Identification");
    int fixed_count = 0;
    
    // For H1 elements with order 1, DOFs coincide with vertices
    // More generally, we need to check all DOFs by going through elements
    
    Array<bool> dof_checked(ndofs);
    dof_checked = false;
    
    int ne = mesh.GetNE();
    for (int e = 0; e < ne; e++)
    {
        const FiniteElement *fe = fes.GetFE(e);
        ElementTransformation *T = mesh.GetElementTransformation(e);
        
        Array<int> vdofs;
        fes.GetElementVDofs(e, vdofs);
        
        // Get DOF coordinates
        const IntegrationRule &nodes = fe->GetNodes();
        
        for (int i = 0; i < nodes.GetNPoints(); i++)
        {
            const IntegrationPoint &ip = nodes.IntPoint(i);
            Vector x(3);
            T->Transform(ip, x);
            
            int dof_idx = vdofs[i];
            if (dof_idx < 0) dof_idx = -1 - dof_idx;  // Handle orientation
            
            if (!dof_checked[dof_idx])
            {
                dof_checked[dof_idx] = true;
                
                if (GEOM.isInFixedZone(x)) {
                    fixed_dofs.Append(dof_idx);
                    fixed_count++;
                }
            }
        }
    }
    
    DEBUG_LOG.log("Total fixed DOFs", fixed_count);
    DEBUG_LOG.log("Fixed DOF percentage", 100.0 * fixed_count / ndofs);
    
    if (fixed_count == 0) {
        DEBUG_LOG.log("WARNING", "No fixed DOFs found! Check geometry!");
        cout << "WARNING: No fixed DOFs identified! Check fixed zone parameters." << endl;
    }
}

// ============================================================================
// GENERATE CYLINDRICAL MESH
// ============================================================================

Mesh* GenerateCylindricalMesh(int nr, int nz, int ntheta, double radius, double height)
{
    DEBUG_LOG.section("Mesh Generation");
    DEBUG_LOG.log("Radial divisions", nr);
    DEBUG_LOG.log("Height divisions", nz);
    DEBUG_LOG.log("Angular divisions", ntheta);
    DEBUG_LOG.log("Cylinder radius", radius);
    DEBUG_LOG.log("Cylinder height", height);
    
    // Create a 3D mesh manually by specifying vertices and hexahedral elements
    int nv = (nr + 1) * ntheta * (nz + 1);
    int ne = nr * ntheta * nz;
    
    Mesh *mesh = new Mesh(3, nv, ne, 0, 3);
    
    // Generate vertices
    for (int k = 0; k <= nz; k++) {
        double z = -height/2.0 + k * height / nz;
        for (int j = 0; j < ntheta; j++) {
            double theta = 2.0 * M_PI * j / ntheta;
            for (int i = 0; i <= nr; i++) {
                double r = i * radius / nr;
                double x = r * cos(theta);
                double y = r * sin(theta);
                mesh->AddVertex(x, y, z);
            }
        }
    }
    
    // Generate hexahedral elements
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ntheta; j++) {
            int j_next = (j + 1) % ntheta;
            for (int i = 0; i < nr; i++) {
                int v0 = k * ntheta * (nr+1) + j * (nr+1) + i;
                int v1 = k * ntheta * (nr+1) + j * (nr+1) + i + 1;
                int v2 = k * ntheta * (nr+1) + j_next * (nr+1) + i + 1;
                int v3 = k * ntheta * (nr+1) + j_next * (nr+1) + i;
                int v4 = (k+1) * ntheta * (nr+1) + j * (nr+1) + i;
                int v5 = (k+1) * ntheta * (nr+1) + j * (nr+1) + i + 1;
                int v6 = (k+1) * ntheta * (nr+1) + j_next * (nr+1) + i + 1;
                int v7 = (k+1) * ntheta * (nr+1) + j_next * (nr+1) + i;
                
                int vertices[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
                mesh->AddHex(vertices, 1);
            }
        }
    }
    
    mesh->FinalizeHexMesh(1, 1, true);
    
    DEBUG_LOG.log("Total vertices", mesh->GetNV());
    DEBUG_LOG.log("Total elements", mesh->GetNE());
    DEBUG_LOG.log("Total boundary elements", mesh->GetNBE());
    
    return mesh;
}

// ============================================================================
// ANALYZE BOUNDARY FACES
// ============================================================================

void AnalyzeBoundaryFaces(Mesh &mesh)
{
    DEBUG_LOG.section("Boundary Face Analysis");
    
    int top_faces = 0, bottom_faces = 0, cylindrical_faces = 0;
    
    for (int i = 0; i < mesh.GetNBE(); i++)
    {
        Array<int> vertices;
        mesh.GetBdrElementVertices(i, vertices);
        
        // Average position of boundary element
        Vector avg_pos(3);
        avg_pos = 0.0;
        for (int j = 0; j < vertices.Size(); j++) {
            const double *v = mesh.GetVertex(vertices[j]);
            avg_pos(0) += v[0];
            avg_pos(1) += v[1];
            avg_pos(2) += v[2];
        }
        avg_pos /= vertices.Size();
        
        if (GEOM.isOnTopFace(avg_pos, 0.05)) top_faces++;
        else if (GEOM.isOnBottomFace(avg_pos, 0.05)) bottom_faces++;
        else if (GEOM.isOnCylindricalSurface(avg_pos, 0.05)) cylindrical_faces++;
    }
    
    DEBUG_LOG.log("Top boundary faces", top_faces);
    DEBUG_LOG.log("Bottom boundary faces", bottom_faces);
    DEBUG_LOG.log("Cylindrical boundary faces", cylindrical_faces);
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    DEBUG_LOG.section("INITIALIZATION");

    // ========================================================================
    // PARAMETERS
    // ========================================================================

    // Geometry
    const double cyl_radius = 0.5;
    const double cyl_height = 2.0;
    const double fixed_zone_frac = 0.05;  // 5% at top and bottom remain solid
    
    GEOM = CylinderGeometry(cyl_radius, cyl_height, 0.0, 0.0, 0.0, fixed_zone_frac);
    
    // Mesh resolution
    const int nr = 6;       // radial divisions
    const int nz = 20;      // height divisions
    const int ntheta = 20;  // angular divisions

    // Optimization parameters
    const double vol_frac = 0.3;
    const double rho_min = 1e-6;
    const int max_iter = 125;
    const int non_penalized_iter = 10;
    const double max_simp_exp = 3.0;
    const double tolerance = 1e-4;
    const double move_limit = 0.1;
    const double r_min = 0.08;

    // Boundary conditions
    const double h_val = 100.0;           // Convection coefficient (W/m²·K)
    const double T_amb = 25.0;            // Ambient temperature (°C)
    const double Q_flux = 5000.0;         // Heat flux on top (W/m²)

    DEBUG_LOG.log("Volume fraction target", vol_frac);
    DEBUG_LOG.log("Convection coefficient", h_val);
    DEBUG_LOG.log("Heat flux", Q_flux);
    DEBUG_LOG.log("Fixed zone fraction", fixed_zone_frac);

    cout << "==================================================" << endl;
    cout << "3D CYLINDRICAL THERMAL TOPOLOGY OPTIMIZATION" << endl;
    cout << "==================================================" << endl;
    cout << "Geometry: Cylinder (r=" << cyl_radius << ", h=" << cyl_height << ")" << endl;
    cout << "Target volume fraction: " << vol_frac << endl;
    cout << "Heat flux on top: " << Q_flux << " W/m²" << endl;
    cout << "Convection on sides/bottom: " << h_val << " W/(m²·K)" << endl;
    cout << "Fixed zones: top/bottom " << fixed_zone_frac*100 << "%" << endl;
    cout << "==================================================" << endl;

    // ========================================================================
    // MESH AND FINITE ELEMENT SPACE
    // ========================================================================

    Mesh *mesh = GenerateCylindricalMesh(nr, nz, ntheta, cyl_radius, cyl_height);
    AnalyzeBoundaryFaces(*mesh);

    H1_FECollection fec(1, mesh->Dimension());
    FiniteElementSpace fes(mesh, &fec);

    cout << "Number of unknowns: " << fes.GetNDofs() << endl;
    DEBUG_LOG.log("DOFs", fes.GetNDofs());

    // ========================================================================
    // IDENTIFY FIXED DOFS (top/bottom zones)
    // ========================================================================

    Array<int> fixed_dofs;
    IdentifyFixedDOFs(*mesh, fes, fixed_dofs);

    // ========================================================================
    // INITIALIZE FIELDS
    // ========================================================================

    GridFunction rho(&fes), rho_filtered(&fes), sensitivity(&fes);
    rho = vol_frac;
    
    // Set fixed zones to 1.0
    for (int i = 0; i < fixed_dofs.Size(); i++) {
        rho(fixed_dofs[i]) = 1.0;
    }
    
    sensitivity = 0.0;

    // ========================================================================
    // SETUP COEFFICIENTS
    // ========================================================================

    TopFaceHeatFluxCoeff flux_coeff(Q_flux);
    CylindricalConvectionCoeff h_coeff(h_val);
    CylindricalConvectionAmbientCoeff hTinf_coeff(h_val, T_amb);

    // ========================================================================
    // OPTIMIZATION LOOP
    // ========================================================================

    double simp_exponent = 1.0;
    int exp_update_counter = 0;
    vector<double> compliance_history;
    double old_compliance = 1e30;

    CGSolver cg_solver;
    cg_solver.SetMaxIter(100000);
    cg_solver.SetRelTol(1e-6);
    cg_solver.SetAbsTol(1e-9);
    cg_solver.SetPrintLevel(-1);

    for (int iter = 0; iter < max_iter; iter++)
    {
        DEBUG_LOG.section("ITERATION " + to_string(iter + 1));
        
        cout << "\n================================================" << endl;
        cout << "Iteration " << iter + 1 << "/" << max_iter;
        cout << " (SIMP p=" << simp_exponent << ")" << endl;
        cout << "================================================" << endl;

        // ====================================================================
        // STEP 1: FILTER DESIGN VARIABLES
        // ====================================================================
        densityFilterPDE(*mesh, rho, rho_filtered, r_min);

        // ====================================================================
        // STEP 2: FORWARD ANALYSIS
        // ====================================================================

        ThermalConductivityCoeff k_coeff(&rho_filtered, simp_exponent, rho_min);

        GridFunction temperature(&fes);
        temperature = T_amb;

        cout << "Running steady-state thermal analysis..." << endl;

        // Stiffness matrix
        BilinearForm k_form(&fes);
        k_form.AddDomainIntegrator(new DiffusionIntegrator(k_coeff));
        k_form.AddBoundaryIntegrator(new BoundaryMassIntegrator(h_coeff));
        k_form.Assemble();
        k_form.Finalize();

        // Right-hand side (heat flux on top + convection)
        LinearForm q_form(&fes);
        q_form.AddBoundaryIntegrator(new BoundaryLFIntegrator(flux_coeff));
        q_form.AddBoundaryIntegrator(new BoundaryLFIntegrator(hTinf_coeff));
        q_form.Assemble();

        // No essential BCs (all natural BCs)
        Array<int> ess_tdof_list;
        OperatorPtr A_bc;
        Vector B, X;

        k_form.FormLinearSystem(ess_tdof_list, temperature, q_form, A_bc, X, B);

        // Solve
        cg_solver.SetOperator(*A_bc);
        X = 0.0;
        cg_solver.Mult(B, X);

        DEBUG_LOG.log("CG iterations", cg_solver.GetNumIterations());
        DEBUG_LOG.log("CG converged", cg_solver.GetConverged());

        if (!cg_solver.GetConverged()) {
            cout << "WARNING: CG did not converge! Iterations: " << cg_solver.GetNumIterations() << endl;
        }

        k_form.RecoverFEMSolution(X, q_form, temperature);

        // Calculate compliance
        double compliance_steady = q_form * temperature;
        compliance_history.push_back(compliance_steady);

        DEBUG_LOG.log("Compliance", compliance_steady);

        // Volume calculation
        double current_vol = 0.0;
        for (int i = 0; i < rho_filtered.Size(); i++)
            current_vol += rho_filtered(i);
        current_vol /= rho_filtered.Size();

        // Temperature statistics
        double max_temp = temperature.Max();
        double min_temp = temperature.Min();
        double avg_temp = 0.0;
        for (int i = 0; i < temperature.Size(); i++)
            avg_temp += temperature(i);
        avg_temp /= temperature.Size();

        DEBUG_LOG.log("Max temperature", max_temp);
        DEBUG_LOG.log("Min temperature", min_temp);
        DEBUG_LOG.log("Avg temperature", avg_temp);
        DEBUG_LOG.log("Volume fraction", current_vol);

        cout << "\nCompliance: " << compliance_steady << endl;
        cout << "Temperature - Min: " << min_temp << ", Max: " << max_temp 
             << ", Avg: " << avg_temp << " °C" << endl;
        cout << "Volume fraction: " << current_vol << endl;

        // ====================================================================
        // STEP 3: SENSITIVITY ANALYSIS
        // ====================================================================

        GridFunction sensitivity_physical(&fes);
        ComputeSteadyStateSensitivities(temperature, rho_filtered,
                                        sensitivity_physical, simp_exponent,
                                        rho_min, &fes);

        densityFilterPDE(*mesh, sensitivity_physical, sensitivity, r_min);

        double min_sens = 1e30, max_sens = -1e30;
        for (int i = 0; i < sensitivity.Size(); i++) {
            min_sens = min(min_sens, sensitivity(i));
            max_sens = max(max_sens, sensitivity(i));
        }
        
        DEBUG_LOG.log("Min sensitivity", min_sens);
        DEBUG_LOG.log("Max sensitivity", max_sens);
        
        cout << "Sensitivity range: [" << min_sens << ", " << max_sens << "]" << endl;

        // ====================================================================
        // STEP 4: OPTIMALITY CRITERIA UPDATE (with fixed zones)
        // ====================================================================

        Vector rho_old = rho;
        double lam_min = 1e-10, lam_max = 1e10;

        for (int lam_iter = 0; lam_iter < 50; lam_iter++)
        {
            double lam_mid = 0.5 * (lam_min + lam_max);
            double vol_sum = 0.0;

            for (int i = 0; i < rho.Size(); i++)
            {
                // Skip fixed DOFs
                bool is_fixed = false;
                for (int j = 0; j < fixed_dofs.Size(); j++) {
                    if (fixed_dofs[j] == i) {
                        is_fixed = true;
                        break;
                    }
                }
                
                if (is_fixed) {
                    vol_sum += 1.0;  // Fixed zones contribute their full volume
                    continue;
                }

                double rho_old_i = rho_old(i);
                double sens_i = sensitivity(i);
                double Be = -sens_i / lam_mid;

                double rho_new_i;
                if (Be <= 0) {
                    rho_new_i = max(rho_min, rho_old_i - move_limit);
                } else {
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
        int updated_dofs = 0;

        for (int i = 0; i < rho.Size(); i++)
        {
            // Skip fixed DOFs
            bool is_fixed = false;
            for (int j = 0; j < fixed_dofs.Size(); j++) {
                if (fixed_dofs[j] == i) {
                    is_fixed = true;
                    break;
                }
            }
            
            if (is_fixed) {
                rho(i) = 1.0;  // Keep fixed zones at full density
                continue;
            }

            double rho_old_i = rho_old(i);
            double sens_i = sensitivity(i);
            double Be = -sens_i / lam_final;

            double rho_new_i;
            if (Be <= 0) {
                rho_new_i = max(rho_min, rho_old_i - move_limit);
            } else {
                rho_new_i = max(rho_min, min(1.0, rho_old_i * sqrt(Be)));
                rho_new_i = max(rho_old_i - move_limit, rho_new_i);
                rho_new_i = min(rho_old_i + move_limit, rho_new_i);
            }

            rho_new_i = max(rho_min, min(1.0, rho_new_i));
            max_change = max(max_change, abs(rho_new_i - rho_old_i));
            rho(i) = rho_new_i;
            updated_dofs++;
        }

        DEBUG_LOG.log("Updated DOFs", updated_dofs);
        DEBUG_LOG.log("Max density change", max_change);
        
        cout << "Max density change: " << max_change << endl;

        // ====================================================================
        // STEP 5: CONTINUATION
        // ====================================================================

        exp_update_counter++;

        if (iter >= non_penalized_iter) {
            if (iter == non_penalized_iter)
                cout << "\n*** Starting penalization ***" << endl;

            if (exp_update_counter >= 5 && simp_exponent < max_simp_exp) {
                simp_exponent = min(simp_exponent * 1.05, max_simp_exp);
                exp_update_counter = 0;
                DEBUG_LOG.log("SIMP exponent updated", simp_exponent);
                cout << "⚡ SIMP exponent increased to: " << simp_exponent << endl;
            }
        }

        // ====================================================================
        // STEP 6: CONVERGENCE CHECK
        // ====================================================================

        int binary_count = 0;
        for (int i = 0; i < rho_filtered.Size(); i++) {
            double rho_val = rho_filtered(i);
            if (rho_val < 0.05 || rho_val > 0.95)
                binary_count++;
        }
        double binary_percent = (double)binary_count / rho_filtered.Size() * 100.0;
        
        DEBUG_LOG.log("Binary design %", binary_percent);
        cout << "Binary design: " << binary_percent << "%" << endl;

        if (iter > 20 && abs(compliance_steady - old_compliance) < tolerance * abs(old_compliance)) {
            cout << "\n✓ Converged after " << iter + 1 << " iterations!" << endl;
            DEBUG_LOG.log("Convergence", "ACHIEVED");
            break;
        }

        old_compliance = compliance_steady;

        // ====================================================================
        // STEP 7: PERIODIC OUTPUT
        // ====================================================================

        if ((iter + 1) % 10 == 0 || iter == 0) {
            string result_dir = "simp_3d_solns/";
            fs::create_directories(result_dir);
            
            string filename = result_dir + "cylinder_iter_" + to_string(iter + 1) + ".gf";
            ofstream sol_ofs(filename.c_str());
            rho_filtered.Save(sol_ofs);
            
            DEBUG_LOG.log("Saved intermediate result", filename);
        }
    }

    // ========================================================================
    // SAVE FINAL RESULTS
    // ========================================================================

    DEBUG_LOG.section("SAVING RESULTS");

    string result_dir = "simp_3d_solns/";
    fs::create_directories(result_dir);

    ofstream mesh_ofs(result_dir + "cylinder_thermal.mesh");
    mesh->Print(mesh_ofs);

    ofstream final_rho(result_dir + "cylinder_thermal_final.gf");
    rho_filtered.Save(final_rho);

    GridFunction rho_binary(&fes);
    for (int i = 0; i < rho.Size(); i++)
        rho_binary(i) = (rho_filtered(i) > 0.5) ? 1.0 : rho_min;

    ofstream binary_rho(result_dir + "cylinder_thermal_binary.gf");
    rho_binary.Save(binary_rho);

    // Save compliance history
    ofstream comp_hist(result_dir + "compliance_history.txt");
    for (size_t i = 0; i < compliance_history.size(); i++) {
        comp_hist << i+1 << " " << compliance_history[i] << endl;
    }
    comp_hist.close();

    DEBUG_LOG.log("Mesh saved", result_dir + "cylinder_thermal.mesh");
    DEBUG_LOG.log("Final solution saved", result_dir + "cylinder_thermal_final.gf");
    DEBUG_LOG.log("Binary solution saved", result_dir + "cylinder_thermal_binary.gf");

    cout << "\n==================================================" << endl;
    cout << "OPTIMIZATION COMPLETE" << endl;
    cout << "==================================================" << endl;
    if (compliance_history.size() > 1) {
        cout << "Compliance improvement: "
             << (compliance_history[0] - compliance_history.back()) / compliance_history[0] * 100.0
             << "%" << endl;
    }
    cout << "Final SIMP exponent: " << simp_exponent << endl;
    cout << "Results saved in: " << result_dir << endl;
    cout << "\nTo visualize:" << endl;
    cout << "  glvis -m " << result_dir << "cylinder_thermal.mesh -g " 
         << result_dir << "cylinder_thermal_final.gf" << endl;
    cout << "==================================================" << endl;

    delete mesh;
    MPI_Finalize();
    return 0;
}