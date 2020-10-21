from minieigen import *
from SPFEMexp import SPFEM
import mgis.behaviour as mgis_bv
import numpy
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
import sys
sys.path.append('/home/ceguo/Downloads/optimesh-master')
import optimesh

intType = mgis_bv.IntegrationType.IntegrationWithElasticOperator
hypo = mgis_bv.Hypothesis.PlaneStrain
behaviour = mgis_bv.load('src/libBehaviour.so','STZBehaviour',hypo)

class Plasticity(object):
    def __init__(self,pts,mater=None,sigma=None,gravity=[0.,0.],charlen=1.):
        """
        :var pts: type numpy ndarray, shape = (npts, 2), points coordinates
        :var mater: type structure, material properties
        :var sigma: type numpy ndarray, initial stress, shape = (4,) or (npts, 4)
        :var gravity: type vector, size = 2, gravity constant
        :var charlen: type double, characteristic spacing
        :var limlen: type double, minimum spacing
        """
        self.__npts = len(pts) # no. of nodes
        self.__pts = pts
        self.__spfem = SPFEM() # SPFEM module with C++ backend and Python binding through compiled shared library
        compliance = 1. / mater['young'] * numpy.array([[1.,-mater['poisson'],-mater['poisson'],0.], \
                                                       [-mater['poisson'],1.,-mater['poisson'],0.], \
                                                       [-mater['poisson'],-mater['poisson'],1.,0.], \
                                                       [0.,0.,0.,numpy.sqrt(2)*(1+mater['poisson'])]])
        if sigma is None:
            sigma = numpy.zeros(4)
        epsilon = numpy.dot(compliance, sigma.T).T
        self.__materData = mgis_bv.MaterialDataManager(behaviour, self.__npts) # material data container
        self.mater = mater
        for s in [self.__materData.s0, self.__materData.s1]: # material initialization
            mgis_bv.setMaterialProperty(s,"YoungModulus",mater['young'])
            mgis_bv.setMaterialProperty(s,"PoissonRatio",mater['poisson'])
            mgis_bv.setMaterialProperty(s,"FrictionCoef",mater['mu'])
            mgis_bv.setMaterialProperty(s,"Cohesion",mater['coh'])
            mgis_bv.setMaterialProperty(s,"GrainDiameter",mater['gs'])
            mgis_bv.setMaterialProperty(s,"EffVolExpCoef",mater['alpha'])
            mgis_bv.setMaterialProperty(s,"STZPlastStrain",mater['eps0'])
            mgis_bv.setMaterialProperty(s,"ExcessVol",mater['vz'])
            mgis_bv.setMaterialProperty(s,"EffVolCapacity",mater['c0'])
            mgis_bv.setMaterialProperty(s,"SteadyCompactivity",mater['chi_inf'])
            mgis_bv.setExternalStateVariable(s,"Temperature",293.15)
            s.internal_state_variables[:,:4] = epsilon
            s.internal_state_variables[:,4] = mater['chi0']
            s.thermodynamic_forces[:] = sigma
        self.__wavespeed = numpy.sqrt(mater['young']/mater['rho']/3./(1-2*mater['poisson']))
        print('wave speed:',self.__wavespeed)
        self.__acc = numpy.zeros(self.__npts*2)
        self.__vel = numpy.zeros(self.__npts*2)
        self.__disp = numpy.zeros(self.__npts*2)
        self.__T = 0. # total elapsed time
        self.__charlen = charlen
        self.__updateMeshBmatrixFint() # update the triangulation and internal force vector
        mass = numpy.multiply(self.__area, mater['rho']) # after initialization, mass will not change to guarantee mass conservation
        self.__fbody = numpy.outer(mass, gravity).reshape(-1) # body force due to gravity
        self.__mass = numpy.repeat(mass, 2)

    def __updateMeshBmatrixFint(self): # private method, not callable outside the class (self-explanatory)
        triangles,area,minAlt = self.__spfem.updateMeshBmat(MatrixX(self.__pts),self.__charlen)
        self.__dt = .05 * minAlt / self.__wavespeed
        self.__triangles, self.__area = numpy.array(triangles), numpy.array(area)
        stress = self.__materData.s1.thermodynamic_forces.copy()
        self.__fint = numpy.array(self.__spfem.calcFint(MatrixX(stress)))

    def __updateStrainStress(self,du): # private method (self-explanatory)
        """
        :var du: nodal displacement increment
        """
        dstrain = numpy.array(self.__spfem.calcStrain(VectorX(du)))
        self.__materData.s1.gradients[:,:] += dstrain
        mgis_bv.integrate(self.__materData, intType, self.__dt, 0, self.__materData.n)
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
                et = seg_vec / seg_len # tangential vector →
                en = numpy.cross([0,0,1], et)[:-1] # normal vector 🡑
                gn = numpy.dot(self.__pts-seg[0], en) # normal gap, negative at contact
                ksi = numpy.dot(self.__pts-seg[0], et) / seg_len # normalized tangential projection
                veln = numpy.dot(vel, en)
                msk = (gn <= 0.) * (ksi >= 0.) * (ksi <= 1.) * (veln < 0.) # where contact takes place
                veln *= msk
                accn = numpy.dot(acc, en) * msk
                acc -= accn.reshape(-1,1) * en # subtracting normal acc at contacts
                velt = numpy.dot(vel, et) * msk
                vel -= veln.reshape(-1, 1) * en # subtracting normal vel at contacts
                dvel = -numpy.minimum(numpy.abs(velt), -mu * veln) * numpy.sign(velt)
                vel += dvel.reshape(-1, 1) * et
                acc += (dvel / self.__dt).reshape(-1, 1) * et                
            self.__acc = acc.reshape(-1)
            self.__vel = vel.reshape(-1)
    """
    def __removePoints(self):
        pts_del = self.__spfem.removeClosePts(MatrixX(self.__pts), self.__charlen / 10.)
        if len(pts_del) > 0:
            print('%d pts removed...'%len(pts_del))
        msk = numpy.ones(self.__npts, dtype=bool)
        msk[pts_del] = False
        self.__pts = self.__pts[msk]
        self.__npts = len(self.__pts)
        
        isv = self.__materData.s1.internal_state_variables.copy() # Elastic strain tensor (4), compactivity (1), equivalent viscoplastic strain (1), viscoplastic strain tensor (4)
        sig = self.__materData.s1.thermodynamic_forces.copy()
        grad = self.__materData.s1.gradients.copy()
        self.__materData = mgis_bv.MaterialDataManager(behaviour, self.__npts)
        for s in [self.__materData.s0, self.__materData.s1]:
            mgis_bv.setMaterialProperty(s,"YoungModulus",self.mater['young'])
            mgis_bv.setMaterialProperty(s,"PoissonRatio",self.mater['poisson'])
            mgis_bv.setMaterialProperty(s,"FrictionCoef",self.mater['mu'])
            mgis_bv.setMaterialProperty(s,"Cohesion",self.mater['coh'])
            mgis_bv.setMaterialProperty(s,"GrainDiameter",self.mater['gs'])
            mgis_bv.setMaterialProperty(s,"EffVolExpCoef",self.mater['alpha'])
            mgis_bv.setMaterialProperty(s,"STZPlastStrain",self.mater['eps0'])
            mgis_bv.setMaterialProperty(s,"ExcessVol",self.mater['vz'])
            mgis_bv.setMaterialProperty(s,"EffVolCapacity",self.mater['c0'])
            mgis_bv.setMaterialProperty(s,"SteadyCompactivity",self.mater['chi_inf'])
            mgis_bv.setExternalStateVariable(s,"Temperature",293.15)
            s.internal_state_variables[:,:] = isv[msk,:]
            s.thermodynamic_forces[:,:] = sig[msk,:]
            s.gradients[:,:] = grad[msk,:]
        
        msk = numpy.repeat(msk, 2)
        self.__acc = self.__acc[msk]
        self.__vel = self.__vel[msk]
        self.__disp = self.__disp[msk]
        self.__fext = self.__fext[msk]
        self.__fbody = self.__fbody[msk]
        self.__dmsk = self.__dmsk[msk]
        self.__dval = self.__dval[msk]
        self.__mass = self.__mass[msk]

    def updatePoints(self):
        '''
        isv = self.__materData.s1.internal_state_variables.copy() # Elastic strain tensor (4), compactivity (1), equivalent viscoplastic strain (1), viscoplastic strain tensor (4)
        sig = self.__materData.s1.thermodynamic_forces.copy()
        grad = self.__materData.s1.gradients.copy()
        acc = self.__acc.reshape(-1,2)
        vel = self.__vel.reshape(-1,2)
        disp = self.__disp.reshape(-1,2)
        fbody = self.__fbody.reshape(-1,2)
        mass = self.__mass.reshape(-1,2)
        var = numpy.column_stack([isv, sig, grad, acc, vel, disp, fbody, mass])
        pts = self.__pts.copy()
        interp = LinearNDInterpolator(self.__pts, var, fill_value=0., rescale=True) # from scipy.interpolate
        '''
        X, cells = optimesh.cpt.fixed_point_uniform(self.__pts, self.__triangles, 0., 100, omega=.8) # mesh optimization with optimesh
        #res = interp(self.__pts)
        #self.__acc = res[:,18:20].reshape(-1)
        #self.__vel = res[:,20:22].reshape(-1)
        #self.__disp = res[:,22:24].reshape(-1)
        #self.__fbody = res[:,24:26].reshape(-1)
        #self.__mass = res[:,26:].reshape(-1)
        #print('max shift:',max(numpy.linalg.norm(self.__pts-pts,axis=1)))
        #print('max vel diff:',max(numpy.linalg.norm(self.__vel.reshape(-1,2)-vel,axis=1)))
        #self.__materData.s1.thermodynamic_forces[:,:] = res[:,10:14]
        #mgis_bv.update(self.__materData)
        '''
        self.__materData = mgis_bv.MaterialDataManager(behaviour, self.__npts)
        for s in [self.__materData.s0, self.__materData.s1]:
            mgis_bv.setMaterialProperty(s,"YoungModulus",self.mater['young'])
            mgis_bv.setMaterialProperty(s,"PoissonRatio",self.mater['poisson'])
            mgis_bv.setMaterialProperty(s,"FrictionCoef",self.mater['mu'])
            mgis_bv.setMaterialProperty(s,"Cohesion",self.mater['coh'])
            mgis_bv.setMaterialProperty(s,"GrainDiameter",self.mater['gs'])
            mgis_bv.setMaterialProperty(s,"EffVolExpCoef",self.mater['alpha'])
            mgis_bv.setMaterialProperty(s,"STZPlastStrain",self.mater['eps0'])
            mgis_bv.setMaterialProperty(s,"ExcessVol",self.mater['vz'])
            mgis_bv.setMaterialProperty(s,"EffVolCapacity",self.mater['c0'])
            mgis_bv.setMaterialProperty(s,"SteadyCompactivity",self.mater['chi_inf'])
            mgis_bv.setExternalStateVariable(s,"Temperature",293.15)
            s.internal_state_variables[:,:] = isv#res[:,:10]
            s.thermodynamic_forces[:,:] = res[:,10:14]
            s.gradients[:,:] = grad#res[:,14:18]
        '''
        self.__updateMeshBmatrixFint()
    """
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
        function to solve the time marching problem using the velocity Verlet scheme
        """
        self.__vel += .5 * self.__acc * self.__dt
        du = self.__vel * self.__dt
        self.__disp += du
        self.__pts += du.reshape(-1,2)
        self.__updateStrainStress(du) # update strain first
        #self.__updatePts()
        self.__updateMeshBmatrixFint() # then update mesh, B-matrix, internal nodal force
        self.__acc = numpy.divide(self.__fext + self.__fbody - self.__fint, self.__mass)
        self.__acc = self.__acc * numpy.logical_not(self.__dmsk)
        self.__vel += .5 * self.__acc * self.__dt
        self.__updateContact()
        self.__T += self.__dt
        return du

    def saveMeshVar(self,name_prefix='',**kwargs): # save mesh and variables in npz file
        filename = name_prefix + 'meshvar.npz'
        numpy.savez(filename, pts=self.__pts, tri=numpy.array(self.__triangles), t=self.__T, **kwargs)

    def getNumPts(self):
        return self.__npts

    def getPoints(self):
        return self.__pts

    def getStress(self):
        return self.__materData.s1.thermodynamic_forces.copy()

    def getStrain(self):
        return self.__materData.s1.gradients.copy()

    def getCompactivity(self):
        return self.__materData.s1.internal_state_variables[:, 4]

    def getPlasticStrain(self):
        return self.__materData.s1.internal_state_variables[:, 5]

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
