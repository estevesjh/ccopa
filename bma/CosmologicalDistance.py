#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
import math
import sys
import logging

# if scipy exists, replace the bad integration method by a scipy function
try:
    import scipy
    import scipy.integrate
except:
    #print "NO scipy found!"
    pass

class CosmologicalDistance:
    """A class to calculate cosmological distances
    The procedure by Hoggs 2000 and Peebles 1993 are followed
    http://arxiv.org/abs/astro-ph/9905116

    The cosmological parameters of interest are
    h0 ~0.7 = dimensionless constant: hubble parameter / (100 (km/s)/Mpc)
    omega_m ~ 0.3 =  matter content of the universe
    omega_l ~ 0.7 dark energy
    omega_k = 1 - omega_m - omega_l = curvature

    Other parameters:
    redshift = the redshift of the source(s) for which distance is 
               being measured
    """

    # matter content of the universe
    omega_m = 0.3
    
    # cosmological constant / lambda 
    omega_l = 0.7
    
    # curvature term
    omega_k = 1 - omega_m - omega_l
    
    # Hubble constant / 100
    h0      = 0.7
    
    # speed of light (km/s)
    c       = 299792.458
    
    # since one cannot realy compare floats with ==, the Universe would
    # never be flat. The Universe is defined flat if
    # omega_m + omega_l - 1 < flat   or equivalently
    # omega_k < flat
    flat    = 0.00005
    
    # relative tolerance of the integration routine
    tolerance = 1e-6
    
    def __init__(self, omega_m=0.3, omega_l=0.7, h0=0.7):

        logging.debug('Starting CosmologicalDistance.__init__()')

        """Initialize the universe parameters
        
        omega_m is the matter content of the Universe (dark and baryonic)
        omega_l is the vacuum/dark energy/lambda component of the Universe
        h0 is the hubble parameter / (100 km/s/Mpc)
        """
        self.set_omega_m(omega_m)
        self.set_omega_l(omega_l)
        omega_k = self.set_omega_k()
        self.h0 = h0

        logging.debug('omega_m = {omega_m}'.format(
            omega_m=omega_m))
        logging.debug('omega_l = {omega_l}'.format(
            omega_l=omega_l))
        logging.debug('omega_m = {omega_m}'.format(
            omega_m=omega_m))
        logging.debug('omega_k = {omega_k}'.format(
            omega_k=omega_k))

	logging.debug('Returning from CosmologicalDistance.__init__()')


    # Helper functions
    
    def integrate(self, func, a, b, TOL=None):
        """Integrate func from a to b with tolerance TOL."""
        
        # if scipy is available, use it
        if sys.modules.has_key('scipy.integrate'):
            return scipy.integrate.quad(func, a, b)

        # else integrate ourselves
        """
        Closed Simpson's rule for 
            \int_a^b f(x) dx
        Divide [a,b] iteratively into h, h/2, h/4, h/8, ... step sizes; and,
        for each step size, evaluate f(x) at a+h, a+3h, a+5h, a+7h, .., b-3h,
        b-h, noting that other points have already been sampled.
        
        At each iteration step, data are sampled only where necessary so that
        the total data is represented by adding sampled points from all
        previous steps:
            step 1:	h	a---------------b
            step 2:	h/2 	a-------^-------b
            step 3:	h/4	a---^-------^---b
            step 4:	h/8	a-^---^---^---^-b
            total:		a-^-^-^-^-^-^-^-b
        So, for step size of h/n, there are n intervals, and the data are
        sampled at the boundaries including the 2 end points.
        
        If old = Trapezoid formula for an old step size 2h, then Trapezoid
        formula for the new step size h is obtained by 
            new = old/2 + h{f(a+h) + f(a+3h) + f(a+5h) +...+ f(b-3h)
                + f(b-h)}
        Also, Simpson formula for the new step size h is given by
            simpson = (4 new - old)/3
        """
        if not TOL: TOL = self.tolerance
        h = b - a
        old2 = old = h * (func(a) + func(b)) / 2.0
        count = 0
        while 1:
            h = h / 2.0
            x, sum = a + h, 0
            while x < b:
                sum = sum + func(x)
                x = x + 2 * h
            new = old / 2.0 + h * sum
            new2 = (4 * new - old) / 3.0
            #if abs(new2 - old2) < TOL * (1 + abs(old2)): return new2
            if abs(new2 - old2) < TOL * (1 + abs(old2)): break
            old = new	# Trapezoid
            old2 = new2	# Simpson
            count = count + 1
        
        return (new2,)

    def newton(self, func, funcd, x, TOL=1e-6):   # f(x)=func(x), f'(x)=funcd(x)
        """
        Ubiquitous Newton-Raphson algorithm for solving
            f(x) = 0
        where a root is repeatedly estimated by
            x = x - f(x)/f'(x)
        until |dx|/(1+|x|) < TOL is achieved.  This termination condition is a
        compromise between 
            |dx| < TOL,  if x is small
            |dx|/|x| < TOL,  if x is large
        """
        f, fd = func(x), funcd(x)
        count = 0
        while 1:
            dx = f / float(fd)
            if abs(dx) < TOL * (1 + abs(x)):
                return x - dx
            x = x - dx
            f, fd = func(x), funcd(x)
            count = count + 1
            #print "newton(%d): x=%s, f(x)=%s" % (count, x, f)
            if count == 1000:
                return x

    # Getters / Setters
    
    def __setattr__(self, attr, val):
        if attr in ['omega_k']:
            # omega_k should not be changed manually
            #raise AttributeError, "Cannot set '%s' attribute" % attr
            self.set_omega_k()
        else:
            self.__dict__[attr] = val
        if attr in ['omega_m', 'omega_l']:
            self.set_omega_k()

    
    def set_omega_m(self, omega_m):
        """Set the matter content of the universe (dark and baryonic)."""
        if (numpy.isreal(omega_m) and numpy.isfinite(omega_m)):
            self.omega_m = omega_m
            self.set_omega_k()
        return self.omega_m

    def set_omega_l(self, omega_l):
        """Set the vacuum/dark energy/lambda content of the universe."""
        if (numpy.isreal(omega_l) and numpy.isfinite(omega_l)):
            self.omega_l = omega_l
            self.set_omega_k()
        return self.omega_l

    def set_omega_k(self):
        """You cannot set the curvature of the universe directly."""
        ok = 1 - self.omega_m - self.omega_l
        if ok < self.flat:
            ok = 0.0
        self.__dict__['omega_k'] = ok
        return self.omega_k
    
    def h0(self, h0=numpy.nan):
        """read or set the Hubble Parameter in units of (100 km/s/Mpc)"""
        if (numpy.isreal(h0) and numpy.isfinite(h0)):
            self.h0 = h0
        return self.h0
    
    def hubble_distance(self):
        """Returns the Hubble distance. Hogg 2000 eqn. 4"""
        hubble_distance = self.c/(100*self.h0)
        return hubble_distance


    # Distance calculation helpers

    def ee(self, redshift):
        """Hogg 2000, equation 14. Peebles 1993, pp 310-321."""
        eer = math.sqrt( self.omega_m*((1+redshift)**3) + \
                         self.omega_k*((1+redshift)**2) + self.omega_l )
        return eer

    def eei(self, redshift):
        """The inverse of ee, used for integration"""
        return 1/self.ee(redshift)
    
    def eeii(self, redshift):
        """The integration of the inverse of ee."""
        inte = self.integrate(self.eei, 0, redshift)[0]
        return inte


    # The actual distances

    def comoving_distance(self, redshift):
        """Returns the comoving distance of an object of given redshift
        Hogg 2000, chapter 5, equation 15."""
        if numpy.isnan(redshift) or redshift < 0:
            return numpy.nan
        
        comoving_distance = self.hubble_distance()*self.eeii(redshift)
        return comoving_distance

    def inverse_comoving_distance(self, comoving_distance):
        """Returns the redshift from the comoving distance."""
        if numpy.isnan(redshift) or redshift < 0:
            return numpy.nan
        
        ztrial = comoving_distance / self.hubble_distance()
        zok = self.newton(
            lambda z: self.eeii(z) - ztrial,
            lambda z: self.eei(z),
            ztrial )
        return zok
        
        """
        from astro.util.CosmologicalDistance import CosmologicalDistance
        universe = CosmologicalDistance()
        z = 0.1
        cdt = universe.hubble_distance()*z
        cd = universe.comoving_distance(0.1)
        z2 = universe.inverse_comoving_distance(cd)
        
        """
        

    def proper_motion_distance(self, redshift):
        """Returns the propermotion distance at distance redshift.
        Hogg 2000, chapter 5, equation 16."""
        
        if numpy.isnan(redshift) or redshift < 0:
            return numpy.nan
        
        comoving_distance = self.comoving_distance(redshift)
        
        # separate 3 cases, flat, hyperbolic or spherical
        # mathematically these 3 definitions are equivalent
        # but we would need complex numbers
        # in the limit OmegaK->0 the latter two limit to the first
        if(abs(self.omega_k) < self.flat):
            # flat/parabolic
            proper_motion_distance = comoving_distance
        elif(self.omega_k > 0):
            # spherical/elliptical
            proper_motion_distance = self.hubble_distance() *  \
                                     math.sinh(math.sqrt(self.omega_k) * \
                                     comoving_distance /       \
                                     self.hubble_distance()) / \
                                     math.sqrt(self.omega_k)
        elif(self.omega_k < 0):
            # hyperbolic
            proper_motion_distance = self.hubble_distance() * \
                math.sin( math.sqrt(abs(self.omega_k)) *     \
                          comoving_distance /                 \
                          self.hubble_distance()              \
                        ) / math.sqrt(abs(self.omega_k))
        else:
            proper_motion_distance = comoving_distance
            # this could never happen

        return proper_motion_distance

    def transverse_comoving_distance(self, redshift, arc):
        """Returns the comoving distance between two points at a
        distance 'redshift' seperated by an angular separation arc
        Hogg 2000, chapter 5."""
        if numpy.isnan(redshift) or redshift < 0:
            return numpy.nan
        
        distance = arc * self.proper_motion_distance(redshift)
        return distance

    def angular_diameter_distance(self, redshift):
        """Returns the angular diameter distance: the ratio of an
        object's physical transverse size to its angular size.
        Hogg 2000, chapter 6, equation 18."""
        if numpy.isnan(redshift) or redshift < 0:
            return numpy.nan
        
        angular_diameter_distance = self.proper_motion_distance(redshift) / \
                                    (1+redshift)
        return angular_diameter_distance

    def luminosity_distance(self, redshift):
        """Returns the luminosity distance.
        Hogg 2000, chapter 7, equation 21."""
        if numpy.isnan(redshift) or redshift < 0:
            return numpy.nan
        
        luminosity_distance = self.proper_motion_distance(redshift) * \
                              (1+redshift)
        return luminosity_distance

    def comoving_volume_element(self, redshift):
        """Returns the comoving volume element in Mpc^3.
        Hogg 2000, chapter 9, equation 28."""
        if numpy.isnan(redshift) or redshift < 0:
            return numpy.nan
        
        comoving_volume_element = self.hubble_distance() * \
                             (1+redshift)**2 *             \
                             self.angular_diameter_distance(redshift)**2 / \
                             self.ee(redshift)
        return comoving_volume_element

    def comoving_volume(self, redshift):
        """Returns the comoving volume within radius redshift
        for the entire sky in Mpc^3."""
        if numpy.isnan(redshift) or redshift < 0:
            return numpy.nan
        
        comoving_volume = 4*math.pi * \
              self.integrate(self.comoving_volume_element, 0, redshift)[0]
        return comoving_volume

    def comoving_volume_slice(self, redshiftmin, redshiftmax, solidangle):
        """Returns the comoving volume in Mpc^3of a slice of angulare size
        'solidangle' (in sterradian) between redshifts 'redshiftmin' and
        'redshiftmax'."""
        if numpy.isnan(redshift) or redshift < 0:
            return numpy.nan
        
        vol1 = self.comoving_volume(redshiftmin)
        vol2 = self.comoving_volume(redshiftmax)
        vol = vol2 - vol1
        vol *= (solidangle/(4*math.pi))
        return vol

    def angular_separation(self, ra0, dec0, ra1, dec1):
        """Calculates the angular separation between two points using
        the haversine formula. Input and output in radians."""
        a1 = math.cos(dec0) * math.cos(dec1) * math.sin( (ra0-ra1)/2 )**2
        a2 = math.sin((dec0-dec1)/2)**2
        sigma = 2*math.asin( math.sqrt( a1+a2 ) )
        return sigma




if(__name__ == '__main__'):
    """Testing cosmological distances,
    this should plot images from Hogg 2000."""
    print("Testing cosmological distances.")
    print("This should plot images from Hogg 2000 if pylab is available.")
    #u = CosmologicalDistance(0.2,0.8,0.7)
    u = CosmologicalDistance(0.3,0.7,0.7)
    z = 1.0
    print("redshift = "+str(z))
    print("comoving distance         = "+str(u.comoving_distance(z)))
    print("proper motion distance    = "+str(u.proper_motion_distance(z)))
    print("angular diameter distance = "+str(u.angular_diameter_distance(z)))
    print("luminosity distance       = "+str(u.luminosity_distance(z)))
    print("comoving volume           = "+str(u.comoving_volume(z)))
    print("comoving volume element / hubble**3 = "
        +str(u.comoving_volume_element(z)/(u.hubble_distance()**3)))

    try:
        import pylab
    except:
        print("no pylab, no plots generated")
    
    if pylab:
    
        universes=[
            CosmologicalDistance( 1.0, 0.0, 0.7), # Einstein de Sitter
            CosmologicalDistance(0.05, 0.0, 0.7), # Low Density
            CosmologicalDistance( 0.2, 0.8, 0.7)  # High Lambda
        ]
    
        redshiftmax = 5.0
        redshifts = numpy.arange(0., redshiftmax, redshiftmax/100.)
        
        dists = []
        for universeCount in range(len(universes)):
            dists.append(numpy.arange(0., redshiftmax, redshiftmax/100.))
        
        print("plotting Proper Motion Distance, Hogg figure 1")
        for universeCount in range(len(universes)):
            for i in range(len(redshifts)):
                dists[universeCount][i] = \
                universes[universeCount].\
                proper_motion_distance(redshifts[i])\
                /universes[universeCount].hubble_distance()
        pylab.plot(redshifts,dists[0],'r-')
        pylab.plot(redshifts,dists[1],'b.')
        pylab.plot(redshifts,dists[2],'g--')
        pylab.ylim(0.0,3.0)
        pylab.xlabel('redshift')
        pylab.ylabel('proper motion distance')
        pylab.show()
    
        print("plotting Angular Diameter Distance, Hogg figure 2")
        for universeCount in range(len(universes)):
            for i in range(len(redshifts)):
                dists[universeCount][i] = \
                  universes[universeCount].\
                  angular_diameter_distance(redshifts[i])\
                  /universes[universeCount].hubble_distance()
        pylab.plot(redshifts,dists[0],'r-')
        pylab.plot(redshifts,dists[1],'b.')
        pylab.plot(redshifts,dists[2],'g--')
        pylab.ylim(0.0,0.5)
        pylab.xlabel('redshift')
        pylab.ylabel('angular diameter distance')
        pylab.show()
    
        print("plotting Luminosity Distance, Hogg figure 3")
        for universeCount in range(len(universes)):
            for i in range(len(redshifts)):
                dists[universeCount][i] =\
                  universes[universeCount].\
                  luminosity_distance(redshifts[i])\
                  /universes[universeCount].hubble_distance()
        pylab.plot(redshifts,dists[0],'r-')
        pylab.plot(redshifts,dists[1],'b.')
        pylab.plot(redshifts,dists[2],'g--')
        pylab.ylim(0.0,16.)
        pylab.xlabel('redshift')
        pylab.ylabel('luminosity diameter distance')
        pylab.show()
    
        print("plotting Comoving Volume Element, Hogg figure 5")
        for universeCount in range(len(universes)):
            for i in range(len(redshifts)):
                dists[universeCount][i] =\
                  universes[universeCount].\
                  comoving_volume_element(redshifts[i])\
                  /(universes[universeCount].hubble_distance()**3)
        pylab.plot(redshifts,dists[0],'r-')
        pylab.plot(redshifts,dists[1],'b.')
        pylab.plot(redshifts,dists[2],'g--')
        pylab.ylim(0.0,1.2)
        pylab.xlabel('redshift')
        pylab.ylabel('comoving volume element')
        pylab.show()
    
