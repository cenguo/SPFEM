#include <boost/python.hpp> //use Boost to interact with Python
#include <vector> //standard vector container
#include <Eigen/Core> //Eigen for matrix operations
#include <Eigen/Sparse>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h> //CGAL
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel            K;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned, K>       Vb;
typedef CGAL::Triangulation_data_structure_2<Vb>                       Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds>                         Delaunay;
typedef K::Point_2                                                     Point_2;
typedef Delaunay::Finite_faces_iterator                                Finite_faces_iterator;

namespace py = boost::python;

class SPFEM {
    private:
        int npts; //number of points (nodes)
        std::vector<std::pair<Point_2,unsigned>> points; //node coordinates
        Eigen::VectorXd area; //nodal area
        std::vector<Eigen::SparseMatrix<double>> bmatrices; //tilde_B

    public:
        SPFEM() {}
        ~SPFEM() {}
        py::tuple updateMeshBmat(const Eigen::MatrixXd& pts, const double charlen);
        Eigen::VectorXd calcFint(const Eigen::MatrixXd& sig);
        py::list calcStrain(const Eigen::VectorXd& du);
};

py::tuple SPFEM::updateMeshBmat(const Eigen::MatrixXd& pts, const double charlen) {
    npts = pts.rows(); //number of points
    area = Eigen::VectorXd::Zero(npts);
    points.clear();
    bmatrices.clear();
    Eigen::SparseMatrix<double> spMat(3,npts*2);
    for(size_t i=0; i<npts; i++) { //convert to CGAL Point_2 type
        points.push_back(std::make_pair(Point_2(pts(i,0),pts(i,1)),i));
        bmatrices.push_back(spMat);
    }
    Delaunay dt(points.begin(),points.end()); //Delaunay triangulation
    Point_2 v[3];
    int ind[3];
    py::list triangles;
    double circumrad,atri,max_edge_len,min_altitude=charlen;
    double edge_len[3];
    for(Finite_faces_iterator fit=dt.finite_faces_begin();fit!=dt.finite_faces_end();fit++) { //loop over all triangles in dt
        for(size_t i=0; i<3; i++) { //loop over vertices of each triangle
            v[i] = fit->vertex(i)->point(); //vertex coordinates
            ind[i] = fit->vertex(i)->info(); //vertex index
        }
        circumrad = CGAL::sqrt(CGAL::squared_radius(v[0],v[1],v[2])); //triangle circumradius
        atri = CGAL::area(v[0],v[1],v[2]); //triangle area
        if(circumrad < charlen && atri > 1.e-10) { //apply alpha shapes
            triangles.append(py::make_tuple(ind[0],ind[1],ind[2]));
            edge_len[0] = CGAL::sqrt(CGAL::squared_distance(v[1],v[2]));
            edge_len[1] = CGAL::sqrt(CGAL::squared_distance(v[2],v[0]));
            edge_len[2] = CGAL::sqrt(CGAL::squared_distance(v[0],v[1]));
            max_edge_len = std::max(std::max(edge_len[0],edge_len[1]),edge_len[2]);
            min_altitude = std::min(min_altitude, 2. * atri / max_edge_len);
            for(size_t i=0; i<3; i++) {
                area[ind[i]] += atri / 3.; //smoothed nodal area
                for(size_t j=0; j<3; j++) {
                    /*bmatrices is a vector of Eigen sparse matrices.
                    The following block finds sum(A_k*B_k) for each node.*/
                    bmatrices[ind[i]].coeffRef(0,ind[j]*2) += (v[(j+1)%3][1] - v[(j+2)%3][1]) / 6.;
                    bmatrices[ind[i]].coeffRef(1,ind[j]*2+1) += (v[(j+2)%3][0] - v[(j+1)%3][0]) / 6.;
                    bmatrices[ind[i]].coeffRef(2,ind[j]*2) += (v[(j+2)%3][0] - v[(j+1)%3][0]) / 6.;
                    bmatrices[ind[i]].coeffRef(2,ind[j]*2+1) += (v[(j+1)%3][1] - v[(j+2)%3][1]) / 6.;
                }
            }
        }
    }
    for(size_t i=0; i<npts; i++) { //tilde_B = sum(A_k*B_k)/node_area
        bmatrices[i] /= (area[i] + 1.e-20);
    }
    return py::make_tuple(triangles,area,min_altitude);
}

Eigen::VectorXd SPFEM::calcFint(const Eigen::MatrixXd& sig) {
    Eigen::VectorXd fint = Eigen::VectorXd::Zero(npts*2);
    Eigen::Vector3d stress;
    for(size_t i=0; i<npts; i++) {
        stress << sig(i,0),sig(i,1),sig(i,3)/std::sqrt(2); //conversion from MFront stress vector
        fint += bmatrices[i].transpose() * stress * area[i];
    }
    return fint;
}

py::list SPFEM::calcStrain(const Eigen::VectorXd& du) {
    Eigen::Vector3d eps;
    py::list eps_lst;
    for(size_t i=0; i<npts; i++) {
        eps = bmatrices[i] * du;
        eps_lst.append(py::make_tuple(eps[0],eps[1],0.,eps[2]/std::sqrt(2))); //strain vector in MFront convention
    }
    return eps_lst;
}

BOOST_PYTHON_MODULE(SPFEMexp) //wrap C++ class as Python library
{
    Py_Initialize();
    py::class_<SPFEM>("SPFEM", py::init<>())
        .def("updateMeshBmat", &SPFEM::updateMeshBmat)
        .def("calcFint", &SPFEM::calcFint)
        .def("calcStrain", &SPFEM::calcStrain)
    ;
}
