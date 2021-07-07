from minieigen import * #Eigen for Python, publically available
from SPFEMexp import SPFEM #import compiled SPFEMexp library
import mgis #MFrontGenericInterfaceSupport
import mgis.behaviour as mgis_bv
import numpy
import optimesh #mesh optimization
from scipy.interpolate import LinearNDInterpolator

pool = mgis.ThreadPool(20) #thread pool for parallel stress integration
IT = mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator
behaviour = mgis_bv.load('src/libBehaviour.so','MisesVocePlasticity',mgis_bv.Hypothesis.PlaneStrain) #plane strain assumption

class spfem_mfront(object):
    def __init__(self,pts,mater,sigma=None,gravity=[0.,0.],charlen=1.):
        """
        pts: type numpy ndarray, shape = (npts, 2), points coordinates
        mater: type structure, material properties
        sigma: type numpy ndarray, initial stress, shape = (4,) or (npts, 4)
        gravity: type vector, size = 2, gravity constant
        charlen: type double, characteristic spacing
        """
        self.__npts = len(pts)
        self.__pts = pts
        self.__spfem = SPFEM() #SPFEM module with C++ backend and Python binding through compiled shared library
        compliance = 1. / mater['young'] * numpy.array([[1.,-mater['poisson'],-mater['poisson'],0.], \
                                                       [-mater['poisson'],1.,-mater['poisson'],0.], \
                                                       [-mater['poisson'],-mater['poisson'],1.,0.], \
                                                       [0.,0.,0.,numpy.sqrt(2)*(1+mater['poisson'])]])
        if sigma is None:
            sigma = numpy.zeros(4)
        epsilon = numpy.dot(compliance, sigma.T).T
        self.__materData = mgis_bv.MaterialDataManager(behaviour, self.__npts) #material data container
        self.mater = mater
        for s in [self.__materData.s0, self.__materData.s1]: #material initialization
            mgis_bv.setMaterialProperty(s,"YoungModulus",mater['young'])
            mgis_bv.setMaterialProperty(s,"PoissonRatio",mater['poisson'])
            mgis_bv.setMaterialProperty(s,"InitYieldStress",mater['tau0'])
            mgis_bv.setMaterialProperty(s,"ResidualYieldStress",mater['taur'])
            mgis_bv.setMaterialProperty(s,"SoftenExponent",mater['b'])
            mgis_bv.setExternalStateVariable(s,"Temperature",293.15)
            s.internal_state_variables[:,:4] = epsilon
            s.thermodynamic_forces[:] = sigma
        self.__charlen = charlen
        self.__dt = self.__updateMeshBmatrixFint()
        self.__wavespeed = numpy.sqrt(mater['young']/mater['rho'])
        self.__acc = numpy.zeros(self.__npts*2) #acceleration
        self.__vel = numpy.zeros(self.__npts*2) #velocity
        self.__disp = numpy.zeros(self.__npts*2) #displacement
        self.__T = 0. #total elapsed time
        mass = numpy.multiply(self.__area, mater['rho']) #after initialization, mass will not change to guarantee mass conservation
        self.__fbody = numpy.outer(mass, gravity).reshape(-1) #gravity force
        self.__mass = numpy.repeat(mass, 2) #nodal mass

    def __updateMeshBmatrixFint(self): #private method, not callable outside the class
        tri,area,minAlt = self.__spfem.updateMeshBmat(MatrixX(self.__pts),self.__charlen)
        self.__triangles, self.__area = numpy.array(tri), numpy.array(area)
        stress = self.__materData.s1.thermodynamic_forces.copy()
        self.__fint = numpy.array(self.__spfem.calcFint(MatrixX(stress)))
        return .9 * minAlt / self.__wavespeed #guarantee CFL condition

    def __updateStrainStress(self,du): #private method
        dstrain = self.__spfem.calcStrain(VectorX(du))
        self.__materData.s1.gradients[:,:] += numpy.array(dstrain)
        mgis_bv.integrate(pool, self.__materData, IT, self.__dt) #parallel stress integration
        mgis_bv.update(self.__materData)

    def __updateContact(self):
        if self.__fricFlag:
            acc = self.__acc.reshape(-1,2)
            vel = self.__vel.reshape(-1,2)
            for seg, mu in zip(self.__fricBound, self.__mu):
                seg_vec = seg[1] - seg[0]
                seg_len = numpy.linalg.norm(seg_vec)
                if seg_len < 1.e-6:
                    continue
                et = seg_vec / seg_len # tangential vector (right)
                en = numpy.cross([0,0,1], et)[:-1] # normal vector (up)
                gn = numpy.dot(self.__pts-seg[0], en) # normal gap, negative at contact
                ksi = numpy.dot(self.__pts-seg[0], et) / seg_len # normalized tangential projection
                veln = numpy.dot(vel, en)
                msk = (gn <= 0.) * (ksi >= 0.) * (ksi <= 1.) * (veln < 0.)# where contact takes place
                accn = numpy.dot(acc, en) * msk
                acc -= accn.reshape(-1,1) * en # subtracting normal acc at contacts
                veln *= msk
                velt = numpy.dot(vel, et) * msk
                vel -= veln.reshape(-1, 1) * en # subtracting normal vel at contacts
                dvel = -numpy.minimum(numpy.abs(velt), -mu * veln) * numpy.sign(velt)
                vel += dvel.reshape(-1, 1) * et
                acc += (dvel / self.__dt).reshape(-1, 1) * et                
            self.__acc = acc.reshape(-1)
            self.__vel = vel.reshape(-1)

    def smoothPoints(self):
        pts, cells = self.__pts.copy(), self.__triangles.copy()
        pts, cells = optimesh.optimize_points_cells(pts, cells, "cpt (fixed-point)", 0.0, 100, omega=.8)
        if not numpy.isnan(pts).any():
            isv = self.__materData.s1.internal_state_variables.copy() # Elastic strain tensor (4), plastic strain (1)
            #sig = self.__materData.s1.thermodynamic_forces.copy() # Stress (4)
            grad = self.__materData.s1.gradients.copy() # Total strain (4)
            acc = self.__acc.reshape(-1,2)
            vel = self.__vel.reshape(-1,2)
            disp = self.__disp.reshape(-1,2)
            var = numpy.column_stack([isv, grad, acc, vel, disp])
            interp = LinearNDInterpolator(self.__pts, var)
            res = interp(pts)
            self.__pts = pts.copy()
            self.__updateMaterialData(res[:,:9])
            self.__acc = res[:,9:11].reshape(-1)
            self.__vel = res[:,11:13].reshape(-1)
            self.__disp = res[:,13:].reshape(-1)
            self.__dt = self.__updateMeshBmatrixFint()

    def __updateMaterialData(self,res):
        self.__materData = mgis_bv.MaterialDataManager(behaviour, self.__npts)
        for s in [self.__materData.s0, self.__materData.s1]: # material initialization
            mgis_bv.setMaterialProperty(s,"YoungModulus",self.mater['young'])
            mgis_bv.setMaterialProperty(s,"PoissonRatio",self.mater['poisson'])
            mgis_bv.setMaterialProperty(s,"InitYieldStress",self.mater['tau0'])
            mgis_bv.setMaterialProperty(s,"ResidualYieldStress",self.mater['taur'])
            mgis_bv.setMaterialProperty(s,"SoftenExponent",self.mater['b'])
            mgis_bv.setExternalStateVariable(s,"Temperature",293.15)
            s.internal_state_variables[:,:] = res[:,:5]
            #s.thermodynamic_forces[:,:] = res[:,5:9]
            s.gradients[:,:] = res[:,5:]
        mgis_bv.integrate(pool, self.__materData, intType, self.__dt)
        mgis_bv.update(self.__materData)

    def initTimeStep(self,Fext=None,Dmsk=None,Dval=None,FricBound=None,mu=None): # for each loading time step, do initialization
        """
        function to initialize for each time step, i.e. to set and update boundary condition
        :var Fext: external boundary traction force
        :var Dmsk: Dirichlet boundary (velocity) condition mask
        :var Dval: Dirichlet boundary (velocity) condition value
        :var FricBound: frictional contact boundary composed of line segments, shape = (nseg, 2, 2)
        :var mu: frictional coefficient at each point
        """
        if not (Fext is None):
            self.__fext = Fext
        else:
            self.__fext = numpy.zeros(self.__npts*2)
        if not (Dmsk is None):
            self.__dmsk = Dmsk
        else:
            self.__dmsk = numpy.zeros(self.__npts*2,dtype=bool)
        if not (Dval is None):
            self.__dval = Dval
        else:
            self.__dval = numpy.zeros(self.__npts*2)
        self.__fricFlag = False
        if not (FricBound is None):
            self.__fricFlag = True
            assert(len(FricBound) == len(mu))
            self.__fricBound = FricBound
            self.__mu = mu
        self.__acc = numpy.divide(self.__fext + self.__fbody - self.__fint, self.__mass)
        self.__acc = self.__acc * numpy.logical_not(self.__dmsk)
        self.__vel = self.__vel * numpy.logical_not(self.__dmsk) + self.__dval * self.__dmsk

    def solve(self):
        """ 
        main function using velocity Verlet scheme to solve one time step
        """
        self.__vel += .5 * self.__acc * self.__dt
        du = self.__vel * self.__dt
        self.__disp += du
        self.__pts += du.reshape(-1,2)
        self.__updateStrainStress(du) #update strain first (USF)
        dt = self.__updateMeshBmatrixFint()
        self.__acc = numpy.divide(self.__fext + self.__fbody - self.__fint, self.__mass)
        self.__acc = self.__acc * numpy.logical_not(self.__dmsk) #comply with boundary condition
        self.__vel += .5 * self.__acc * self.__dt
        self.__updateContact() #if there is any contact
        self.__T += self.__dt #total elapsed time
        self.__dt = dt

    def saveMeshVar(self,name_prefix='',**kwargs): #save mesh and variables in npz file for post-processing
        filename = name_prefix + 'meshvar.npz'
        numpy.savez(filename, pts=self.__pts, tri=self.__triangles, t=self.__T, **kwargs)

    def getNumPts(self):
        return self.__npts

    def getPoints(self):
        return self.__pts

    def getStress(self):
        return self.__materData.s1.thermodynamic_forces.copy()

    def getStrain(self):
        return self.__materData.s1.gradients.copy()

    def getPlasticStrain(self):
        return self.__materData.s1.internal_state_variables[:, 4]

    def getTimeStepSize(self):
        return self.__dt

    def getNodalInternalForce(self):
        return self.__fint

    def getNodalAcc(self):
        return self.__acc

    def getNodalVel(self):
        return self.__vel

    def getNodalDisp(self):
        return self.__disp

    def setGravity(self,gravity=[0.,0.]):
        assert(len(gravity) == 2)
        mass = self.__mass[::2]
        self.__fbody = numpy.outer(mass, gravity).reshape(-1)

    def resetAllZero(self):
        self.__disp = numpy.zeros(self.__npts*2)
        self.__vel = numpy.zeros(self.__npts*2)
        self.__acc = numpy.zeros(self.__npts*2)
        self.__T = 0.
