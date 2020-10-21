#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <vector>
#include <Eigen/LU>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel            K;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned, K>       Vb;
typedef CGAL::Triangulation_data_structure_2<Vb>                       Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds>                         Delaunay;
typedef K::Point_2                                                     Point_2;
typedef Delaunay::Finite_edges_iterator                                Finite_edges_iterator;
typedef Delaunay::Finite_faces_iterator                                Finite_faces_iterator;

namespace py = boost::python;
namespace np = boost::python::numpy;

class SPFEM {
    private:
        int npts;
        std::vector<std::pair<Point_2,unsigned>> points;
        Eigen::VectorXd area;
        std::vector<Eigen::SparseMatrix<double>> bmatrices;

    public:
        SPFEM() {}
        ~SPFEM() {}
        py::list removeClosePts(const Eigen::MatrixXd& pts, const double limit);
        py::tuple updateMeshBmat(const Eigen::MatrixXd& pts, const double charlen);
	    Eigen::VectorXd calcFint(const Eigen::MatrixXd& sig);
        py::list calcStrain(const Eigen::VectorXd& du);
};

py::list SPFEM::removeClosePts(const Eigen::MatrixXd& pts, const double limit) {
    npts = pts.rows();
    points.clear();
    for(size_t i=0; i<npts; i++)
        points.push_back(std::make_pair(Point_2(pts(i,0),pts(i,1)),i));
    Delaunay dt(points.begin(),points.end());
    py::list pts_del;
    double edge_len;
    for(Finite_edges_iterator eit=dt.finite_edges_begin(); eit!=dt.finite_edges_end(); eit++) {
        edge_len = CGAL::sqrt(CGAL::squared_distance(eit->first->vertex(Delaunay::cw(eit->second))->point(), 
                                                     eit->first->vertex(Delaunay::ccw(eit->second))->point()));
        if(edge_len < limit)
            pts_del.append(eit->first->vertex(Delaunay::cw(eit->second))->info());
    }
    return pts_del;
}

py::tuple SPFEM::updateMeshBmat(const Eigen::MatrixXd& pts, const double charlen) {
    npts = pts.rows();
    area = Eigen::VectorXd::Zero(npts);
    points.clear();
    bmatrices.clear();
    Eigen::SparseMatrix<double> spMat(3,npts*2);
    for(size_t i=0; i<npts; i++) {
        points.push_back(std::make_pair(Point_2(pts(i,0),pts(i,1)),i));
        bmatrices.push_back(spMat);
    }
    Delaunay dt(points.begin(),points.end());
    Point_2 v[3];
    int ind[3];
    py::list triangles;
    double circumrad,atri,max_edge_len,min_altitude=charlen;
    double edge_len[3];
    for(Finite_faces_iterator fit=dt.finite_faces_begin();fit!=dt.finite_faces_end();fit++) {
        for(size_t i=0; i<3; i++) {
            v[i] = fit->vertex(i)->point();
            ind[i] = fit->vertex(i)->info();
        }
        circumrad = CGAL::sqrt(CGAL::squared_radius(v[0],v[1],v[2]));
        atri = CGAL::area(v[0],v[1],v[2]);
        if(circumrad < charlen && atri > 1.e-7) {
            triangles.append(py::make_tuple(ind[0],ind[1],ind[2]));
            edge_len[0] = CGAL::sqrt(CGAL::squared_distance(v[1],v[2]));
            edge_len[1] = CGAL::sqrt(CGAL::squared_distance(v[2],v[0]));
            edge_len[2] = CGAL::sqrt(CGAL::squared_distance(v[0],v[1]));
            max_edge_len = std::max(std::max(edge_len[0],edge_len[1]),edge_len[2]);
            min_altitude = std::min(min_altitude, 2. * atri / max_edge_len);
            for(size_t i=0; i<3; i++) {
                area[ind[i]] += atri / 3.;
                for(size_t j=0; j<3; j++) {
                    bmatrices[ind[i]].coeffRef(0,ind[j]*2) += (v[(j+1)%3][1] - v[(j+2)%3][1]) / 6.;
                    bmatrices[ind[i]].coeffRef(1,ind[j]*2+1) += (v[(j+2)%3][0] - v[(j+1)%3][0]) / 6.;
                    bmatrices[ind[i]].coeffRef(2,ind[j]*2) += (v[(j+2)%3][0] - v[(j+1)%3][0]) / 6.;
                    bmatrices[ind[i]].coeffRef(2,ind[j]*2+1) += (v[(j+1)%3][1] - v[(j+2)%3][1]) / 6.;
                }
            }
        }
    }
    for(size_t i=0; i<npts; i++)
        bmatrices[i] /= (area[i] + 1.e-20);
    return py::make_tuple(triangles,area,min_altitude);
}

Eigen::VectorXd SPFEM::calcFint(const Eigen::MatrixXd& sig) {
    Eigen::VectorXd fint = Eigen::VectorXd::Zero(npts*2);
    Eigen::Vector3d stress;
    for(size_t i=0; i<npts; i++) {
        stress << sig(i,0),sig(i,1),sig(i,3)/std::sqrt(2);
        fint += bmatrices[i].transpose() * stress * area[i];
    }
    return fint;
}

py::list SPFEM::calcStrain(const Eigen::VectorXd& du) {
    Eigen::Vector3d st;
    py::list ret;
    for(size_t i=0; i<npts; i++) {
        st = bmatrices[i] * du;
        ret.append(py::make_tuple(st[0],st[1],0.,st[2]/std::sqrt(2)));
    }
    return ret;
}

BOOST_PYTHON_MODULE(SPFEMexp)
{
    Py_Initialize();
    np::initialize();
    py::class_<SPFEM>("SPFEM", py::init<>())
        .def("removeClosePts", &SPFEM::removeClosePts)
        .def("updateMeshBmat", &SPFEM::updateMeshBmat)
        .def("calcFint", &SPFEM::calcFint)
        .def("calcStrain", &SPFEM::calcStrain)
    ;
}
