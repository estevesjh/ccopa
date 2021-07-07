import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict

class BayesianProbability:
    ''' Assign Membership Probabilities For Copacabana Galaxies

    input: a, b, gal
    output: collection('prior','marginal','pdfs','pdfs_field','Pmem','Pr','Pz','Pc')

    a: Estimated number of galaxies inside the cluster area
    b: Estimated number of field galaxies inside the cluster area
    gal: table of galaxies, columns: pdf,pdfr,pdfz,pdfc
    '''
    def __init__(self, a, b, r2=1):
        self.alpha = np.where(a<=0.,.1,a)
        self.beta  = b
        self.r2    = r2
        self.betaDist = scipy.stats.beta(self.alpha, self.beta)

        self.prob  = defaultdict(dict)
        # ['prior','marginal','pdfs','pdfs_field','Pmem','Pr','Pz','Pc']

        ## init the prior
        self.compute_flat_prior()

    def load_pdfs(self,gal):
        zcls = np.array(gal['redshift']).copy()
        
        self.pdfz = np.array(gal['pdfz'][:]).copy()
        self.pdfr = np.array(gal['pdfr'][:]).copy()
        self.pdfc = get_color_pdf(zcls,gal['pdfc'][:]).copy()
        
        self.pdfzf= np.array(gal['pdfz_bkg'][:]).copy()
        self.pdfrf= np.array(gal['pdfr_bkg'][:]).copy()*(np.pi*self.r2**2)
        self.pdfcf= get_color_pdf(zcls,gal['pdfc_bkg'][:]).copy()
        
        self.pdf = filter_3pdf(self.pdfz,self.pdfc,self.pdfr)
        self.pdff= filter_3pdf(self.pdfzf,self.pdfcf,self.pdfrf)
        
    def compute_flat_prior(self):
        pm0 = self.betaDist.mean()
        priors = {'flat':pm0,'beta':None}
        self.prob['prior'] = priors

    def compute_marginal(self,name,prior_type):
        pm0 = self.prob['prior'][prior_type]
        pf,pff = self.pick_pdf(name)
        res    = pf*pm0 + pff*(1-pm0)
        self.prob['marginal'][name] = res.copy()
    
    def compute_prob(self,label,name,prior_type,eps=1e-9):
        pm0 = self.prob['prior'][prior_type]
        pdf,pdff = self.pick_pdf(name)
        mar = pdf*pm0 + pdff*(1-pm0)
        res = pdf*pm0/(mar+eps)
        pmem= filter_prob(res.copy())
        self.prob[label][prior_type] = pmem.copy()
    
    def compute_likelihood(self,nbins=40):
        self.pvec = np.linspace(0.,1.,1000)
        pm0   = self.prob['prior']['flat']
        radii = np.linspace(0,self.r2,100)
        pdf   = radial_pdf(radii,self.r2)
        mar   = pdf*pm0+(1-pm0)*(np.pi*self.r2**2)
        p0 = pdf*pm0/(mar+1e-9)
        self.likelihood = get_likelihood(self.pvec,p0,pdf,nbins=nbins)
    
    def compute_beta_prior(self):
        prior = join_pdfs(self.pvec,self.likelihood,self.betaDist.pdf(self.pvec))
        pm0   = mean_pdf(self.pvec,prior)

        self.prob['prior']['beta'] = pm0
        self.prior = prior

    def pick_pdf(self,name):
        if name=='pdfr':
            return [self.pdfr,self.pdfrf]
        
        elif name=='pdfc':
            return [self.pdfc,self.pdfcf]

        elif name=='pdfz':
            return [self.pdfz,self.pdfzf]
        
        elif name=='pdf':
            return [self.pdf,self.pdff]
        else:
            print('error')

    def assign_prob(self,name,label,prior_type):
        self.compute_marginal(name,prior_type)
        self.compute_prob(label,name,prior_type)
    
    def assign_prob_old(self,name,label):
        pdf, pdff = self.pick_pdf(name)
        if name!='Pmem':
            res = doProb(pdf, pdff, self.alpha, self.beta, normed=False)
        else:
            res = doProb(pdf, pdff, self.alpha, self.beta, normed=False)

        self.prob[label]['old'] = res.copy()
    
    def assign_probabilities(self,gal):
        self.names = ['pdf','pdfr','pdfz','pdfc']
        self.labels= ['Pmem','Pr','Pz','Pc']

        ## loading pdfs
        self.load_pdfs(gal)

        ## memb. prob. flat prior
        for label,name in zip(self.labels,self.names):
            self.assign_prob(name,label,'flat')

        ## estimating beta prior
        self.compute_likelihood()
        self.compute_beta_prior()
        
        ## memb. prob. beta prior
        for label,name in zip(self.labels,self.names):
            self.assign_prob(name,label,'beta')

        ## for check purposes
        ## old probabilities definition
        for label,name in zip(self.labels,self.names):
            self.assign_prob_old(name,label)
    
    def compute_ngals(self):
        for col in ['flat','beta','old']:
            self.prob['Ngals'][col] = np.nansum(self.prob['Pmem'][col][:]).copy()

    def plot_prior_distribution(self):
        plt.plot(self.pvec,self.betaDist.pdf(self.pvec),'b',label='Beta')
        plt.plot(self.pvec,self.likelihood,'k',label='P(model|member)')
        plt.plot(self.pvec,self.prior,'r',label='Beta*P(model|member)')
        plt.axvline(self.prob['prior']['flat'],color='b',ls='--')
        plt.axvline(self.prob['prior']['beta'],color='r',ls='--')
        plt.legend()
        plt.xlabel("Prior")
        plt.ylabel('Density')
        pass

def interpData(x,y,x_new):
    out = interp1d(x, y, kind='linear', fill_value='extrapolate')
    return out(x_new)

def join_pdfs(x,pdf1,pdf2,eps=1e-9):
    prod = pdf1*pdf2
    norm = scipy.integrate.simps(prod,x=x)
    return prod/(norm+eps)

def mean_pdf(x,pdf):
    xmean = scipy.integrate.simps(x*pdf,x=x)
    return xmean

def get_likelihood(xnew,x,pdf,eps=1e-9,nbins=20):
    _,xb,pdfb = bin_data(x,pdf,nbins=nbins)
    like = interpData(xb,pdfb,xnew)
    like = np.where(np.isnan(like),0,like)
    norm = scipy.integrate.simps(like,x=xnew)
    return like/(norm+eps)

def bin_data(x,y,nbins=20):
    xbins = np.linspace(0,1.,nbins+1)#np.percentile(x,np.linspace(0,100,nbins+1))
    xmean = np.zeros_like(xbins[1:])
    ymean = np.zeros_like(xbins[1:])
    
    i=0
    for xl,xh in zip(xbins[:-1],xbins[1:]):
        w, = np.where((x>xl)&(x<=xh))
        if w.size>0:
            xmean[i] = np.nanmean(x[w])
            ymean[i] = np.nanmean(y[w])
        i+=1
    return xbins,xmean,ymean

def filter_prob(x,val=0):
    x[np.isnan(x)]=0
    x = np.where(x<0,0.,x)
    x = np.where(x>1,1.,x)
    return x
def filter_pdf(pdf):
    pdf = np.where(pdf>1e6,1e6,pdf)
    return pdf

def filter_3pdf(x,y,z,norm=True):
    norm = (np.sum(x)+np.sum(y)+np.sum(z))
    norm/= (np.sum(x)*np.sum(y)*np.sum(z))
    
    pdf = x*y*z
    if norm:
        pdf*= norm
        
    pdf = filter_pdf(pdf)
    return pdf

## to do: smooth the color pdfs 
## https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
def get_color_pdf(zcls,pdfc):
    return np.array(np.where(zcls<0.35,pdfc[:,0],pdfc[:,2]))
#     return np.ones_like(pdfc[:,0])

def profileNFW(R,R200,c=3):
    #Radial NFW profile implementation. Takes array of radii, value of R200,
    #and NFW concentration parameter (set to 3 by default)
    if R200>0:
        Rs=float(R200)/c
        r=R/Rs
        r=np.where(np.logical_or(r<=1e-5,r==1.),r+0.001,r)
        pre=1./((r**2)-1)
        arctan_coeff=2./(np.sqrt(np.abs(r**2-1)))
        arctan_arg=np.sqrt(np.abs((r-1)/(r+1)))
        sigma=np.where(r>1,pre*(1-arctan_coeff*np.arctan(arctan_arg)),pre*(1-arctan_coeff*np.arctanh(arctan_arg)))
        
        return sigma*2*Rs
    
    else:
        bogusval=-99.*np.ones_like(R)
        return bogusval

def radial_pdf(radii,R200,c=3.5,rc=0.2):
    radii   = np.where(radii<rc/2.,rc/2.,radii)
    density = profileNFW(radii,R200,c=c) ## without norm
    density = np.where(radii<rc,np.mean(density[radii<rc]), density)
    return density

### To Remove
def doProb(Pgals,Pbkg,Ngals,Nbkg, normed=True,eps=1e-12):
    ratio = (Ngals+Nbkg)/( np.sum(Ngals*Pgals) + np.sum(Nbkg*Pbkg) )
    Pgals *= ratio
    Pbkg  *= ratio
    
    if normed:
        Pgals /= (Pgals.sum()+eps)
        Pbkg  /= (Pbkg.sum()+eps)
    
    prob = (Ngals*Pgals)/(Ngals*Pgals+Nbkg*Pbkg+eps)
    # prob[np.isnan(prob)] = 0.

    prob = np.where(prob>1,1.,prob)
    prob = np.where(prob<0.,0.,prob)
    return prob

def chunks(ids1, ids2):
    """Yield successive n-sized chunks from data"""
    for id in ids2:
        w, = np.where( ids1==id )
        yield w

def set_new_columns(table,columns,val=0.):
    for col in columns:
        table[col] = val
    return table

### Assign Probs
def main(cat,gal):
    ncls    = len(cat)

    r200 = np.array(cat['R200']).copy()
    area = np.array(np.pi*r200**2).copy()

    Nc = cat['Norm']
    Nf = cat['Nbkg']*area

    alpha = np.array(Nc).copy()
    beta  = np.array(Nf).copy()

    cids = np.array(cat['CID'])
    gids = np.array(gal['CID'])
    r200 = np.array(cat['R200'])
    keys = list(chunks(gids,cids))

    ## compute the new probabilities
    res = []
    for i in range(ncls):
        print('alpha,beta: %.2f ,%.2f'%(alpha[i],beta[i]))
        b = BayesianProbability(alpha[i],beta[i],r2=r200[i])
        b.assign_probabilities(gal[keys[i]])
        b.compute_ngals()
        
        res.append(b.prob)
        del b
    
    ## assign into the table
    ## assign the new probabilities
    gal = set_new_columns(gal,['Pmem','Pr','Pz','Pc'],val=0.)
    #cat = set_new_columns(cat,['Ngals_%s'%c for c in columns],val=-99.)

    for i in range(ncls):
        for col in ['Pmem','Pr','Pz','Pc']:
            gal[col][keys[i]]  = res[i][col]['beta'][:]
            gal[col+'_flat'][keys[i]]  = res[i][col]['flat'][:]
            gal[col+'_old'][keys[i]]  = res[i][col]['old'][:]
            #cat['Ngals'][i]    = res[i]['Ngals']['beta']


###################### Rewrite Get PDF
def get_frequency(pdf,eps=1e-12):
    pdfn = np.where(pdf<0.,0.,pdf)
    norm = np.sum(pdfn)
    if norm>0.: pdfn = 100*pdfn/norm
    return pdfn

def getPDFs(gal,galIndices,vec_list,pdf_list,nbkg,mag_pdf=False):
    rvec, zvec, cvec, mvec = vec_list
    pdfr, pdfz, pdfc, pdfm = pdf_list

    gal = set_new_columns(gal,['pdf','pdfr','pdfz','pdfm','norm'],val=0.)
    gal = set_new_columns(gal,['pdf_bkg','pdfr_bkg','pdfz_bkg','pdfm_bkg'],val=0.)

    gal['pdfc'] = np.zeros_like(gal['color'])
    gal['pdfc_bkg'] = gal['pdfc']

    for i, idx in enumerate(galIndices):
        nb = nbkg[i]
        ggal = gal[idx] ## group gal

        ## getting pdfs for a given cluster i
        pdfri, pdfr_cfi  = pdfr[0][i], pdfr[1][i]
        pdfzi, pdfzi_bkg = pdfz[0][i], pdfz[2][i]
        pdfci, pdfci_bkg = pdfc[0][i], pdfc[2][i]
        pdfmi, pdfmi_bkg = pdfm[0][i], pdfm[2][i]

        ## setting galaxies variable columns
        r2 = ggal['R200'] 
        radii = ggal['R']
        zgal  = ggal['z']
        color5 = ggal['color']
        mag = ggal['dmag']
        areag = np.pi*r2**2

        radi2 = 0.25*(np.trunc(radii/0.25)+1) ## bins with 0.125 x R200 width
        # areag = np.pi*radi2**2#((radi2+0.25)**2-radi2**2)

        gal['pdfr'][idx] = interpData(rvec,get_frequency(pdfri),radii,extrapolate=True)*areag
        gal['pdfz'][idx] = interpData(zvec,get_frequency(pdfzi),zgal)
        gal['pdfm'][idx] = interpData(mvec,get_frequency(pdfmi),mag)

        gal['pdfr_bkg'][idx] = np.ones_like(radi2)#/areag #interpData(rvec,np.ones_like(gal['pdfr'][idx]),radii)
        gal['pdfz_bkg'][idx] = interpData(zvec,get_frequency(pdfzi_bkg),zgal)
        gal['pdfm_bkg'][idx] = interpData(mvec,get_frequency(pdfmi_bkg),mag)

        for j in range(5):
            gal['pdfc'][idx,j] = interpData(cvec,get_frequency(pdfci[:,j]),color5[:,j])
            gal['pdfc_bkg'][idx,j] = interpData(cvec,get_frequency(pdfci_bkg[:,j]),color5[:,j])
        
        gal['pdf'][idx] = gal['pdfr'][idx]*gal['pdfz'][idx]
        gal['pdf_bkg'][idx] = gal['pdfr_bkg'][idx]*gal['pdfz_bkg'][idx]
        
        if mag_pdf:
            gal['pdf'][idx] *= gal['pdfm'][idx]
            gal['pdf_bkg'][idx] *= gal['pdfm_bkg'][idx]

        # gal['pdfc'][idx,:] = np.where(gal['pdfc'][idx,:]<0.,0.,gal['pdfc'][idx,:])
        # gal['pdfc_bkg'][idx,:] = np.where(gal['pdfc_bkg'][idx,:]<0.,0.,gal['pdfc_bkg'][idx,:])

        # colors: (g-r),(g-i),(r-i),(r-z),(i-z)
        # pick_colors = [0,1,2,3,4]
        # for j in pick_colors:
        #     gal['pdf'][idx] *= gal['pdfc'][idx,j]
        #     gal['pdf_bkg'][idx] *= gal['pdfc_bkg'][idx,j]

        pdfc = np.where(zcls<0.35, gal['pdfc'][:,0], gal['pdfc'][:,2])
        pdfc_bkg = np.where(zcls<0.35, gal['pdfc_bkg'][:,0], gal['pdfc_bkg'][:,2])


        gal['pdf'][idx] *= np.mean(gal['pdfc'][idx],axis=1)
        gal['pdf_bkg'][idx] *= np.mean(gal['pdfc_bkg'][idx],axis=1)
        
        ng_profile = interpData(rvec,pdfr_cfi,radi2,extrapolate=True)
        gal['norm'][idx] = (ng_profile - nb*areag)#/nb

    gal['pdf'] = np.where(gal['pdf']<0.,0.,gal['pdf'])
    gal['pdfr'] = np.where(gal['pdfr']<0.,0.,gal['pdfr'])
    gal['pdfz'] = np.where(gal['pdfz']<0.,0.,gal['pdfz'])
    gal['pdfc'] = np.where(gal['pdfc']<0.,0.,gal['pdfc'])
    gal['pdfm'] = np.where(gal['pdfm']<0.,0.,gal['pdfm'])

    # gal['pdfc'] = np.where(gal['pdfc']<1e-4,0.,gal['pdfc'])
    # gal['pdfc_bkg'] = np.where(gal['pdfc_bkg']<1e-4,0.,gal['pdfc_bkg'])

    return gal

def pick_pdf_color():
