#
# Class MatEl for matrix element files.
#

"""
A class for entire matrix-element files and related functions.
Individual operator matrices are class QCOpMat (see QCOpMat.py)
and low-level I/O is done via the qcmatrixio module (see
qcmatrixio.F).

Class
-----

MatEl -- a container for all the data in a matrix element file.

Properties
----------

unit    -- return the fortran unit to be used while reading or writing.

matlist -- return a list of the names of the operator matrices in the
           object.  Names are the label strings read from a file;
           standard names are in QCMatEl.mat_names.

debug   -- True to turn on extra printing.

scalars -- Array of scalar result values.

Methods
-------

MatEl (debug=False,file=None):
  Create an object, setting the debug flag.  If the file name is given,
  the file is read into the object; otherwise the object is empty
  except for default header parameters.  Any additional keyword arguments
  are passed to MatEl.read if the file is specified.

addobj (obj):
  Add obj (a QCOpMat object) to the set.  The name of the object
  is stored in upper case and synonyms from mat_names_synonyms
  are converted to their full versions.

delobj (name):
  Delete the named operator object if it is present.

scalar (name,*val):
  return/set-and-return one of the scalars, named in QCMatEl.scalar_names
  or QCMatEl.scalar_names_synonyms

set_scalars (name=value,...):
  set several scalar values at once.  the versions of the names in
  scalar_names_synonyms, which have no blanks or punctuation, are
  useful here because all are valid keywords.

read (fname,check_status=False):
  open the matrix element file named fname and load its contents.  If
  check_status is true, then the job status scalar is checked for
  the successful completion flag (1.0).

write (fname):
  open the matrix element file named fname and store to it.

print(doinp=False,**kwargs):
  Print the object and its contents.  Optional keyword arguments
  are passed to print().  This is also called indirectly by
  actions that access the __str__ method.  doinp indicates whether
  the data should be printed in a form useful in python code.

def load_head (self,title="No title",natoms=None,nbasis=0,nbsuse=None,
  icharg=0,multip=1,ne=None,iopcl=0,icgu=None,nfc=0,nfv=0,itran=0,
  ian=None,iattyp=None,atmchg=None,atznuc=None,c=None,atmwgt=None,
  labfil="Gaussian matrix elements",progversion="QCMatEl.py 1.0"):
                 
  update an object with new values for the header information.  the
  remaining header elements, such as the basis set dimensions, are
  reinitialized, so this call should preceed setting new values for
  scalars, the basis set, and storing matrices.

update (matfi=None, matfo=None, check_status=True, doinit=False, **kwargs):

  update the object contents by running a gaussian job and retrieving
  results.  keyword arguments are passed on to makegauinp through rungau
  and specify what is to be computed.  temporary files are used for the
  matrix element input and output files unless they are named.
  check_status indicates whether the successful job completion flag
  in the output matrix element file is to be checked, and IOError
  signalled if the job failed.  if doinit is true then the object
  is reinitialized, so that only the contents of the output matrix
  element file from the calculation will be present; otherwise the
  new results are merged into the old object, leaving any existing data
  not changed by the new calculation as is.

  returns (true,None,None) if the calculation was successful, or
  (false,input,output) with the names of the text input
  and output files if the calculation failed.

Other functions
---------------

DimTuple (n1,n2,n3,n4,n5):
  return a tuple of the specified dimensions, omitting 1s and 0s.

optfile (namei,suffix="",retfd=True):
  return a open file object (retfd True) or the name for a file
  with an optional name.

makegauinp (matfi, matfo, tinput=None, dofock=False, motran=None,
            aotype=None, window=None, miscroute=None, model="HF",
            basis="ChkBasis",program="g16",progargs=[]):

  generate a gaussian input file for a job which takes input from one
  matrix element file and returns results in another.  returns the name
  of the generated file, which is tinput if this was specified or else
  a suitable temporary file and also an updated list of command line
  arguments.

  dofock is true if orbitals have been provided in the input file and
  the fock matrix implied by their density is to be computed, or
  "density" if the density provided is to be used as-is, "SCF" to
  perform a full SCF calculation starting from a generated initial
  guess, and "SCFREAD" to start an SCF calculation from the orbitals
  in the file.

  aotype is "regular", "raf1", "raf2", or "raf3" to request AO 2e-integrals.

  motran is "full" or "partial" to request a full or partial integral
  transformation.  closed-shell full transformations produce one M^4/8
  array, where M=number of active orbitals, closed-shell partial
  transformations produce one (M^2/2,M,O) array where O is the number
  of active occupieds.  open-shell full transformation produces
  three arrays:  M^4/8 AA, (M^2/2,M^2/2) BA and M^4/8 BB.  open-shell
  partial transformation produces four arrays:  (M^2/2,M,OA) AA,
  (M^2/2,M,OB) AB, (M^2/2,M,OA) BA, and (M^2/2,M,OB) BB, where OA
  and OB are the numbers of active alpha and beta electrons.

  window specifies the active orbitals for a transformation.  The
  default is frozen-core, with the usual (largest) definition of
  the core.  Other values are "full" for all orbitals, "frzngc"
  to freeze the largest noble-gas core, "frzingc" to freeze the
  second-largest noble-gas core, and a tuple (m,n) with m and n
  positive to use orbitals m through n, or -m to skip the first m
  occupieds and -n to skip the lowest n virtuals.

  model is the Hamiltonian to be used if a Fock matrix is formed.

  basis is the basis set if this is not to be read from the input files.

  miscroute is other keywords to include in the gaussian route
  section.

  The molecule specification and basis are taken from the input
  matrix element file, as is the specification for spin restricted
  or unrestricted, complex orbitals, etc.

rungau (matfi, matfo, program="g16", progargs=[], debug=False,
        toutput=None, **kwargs):
  run gaussian given the named matrix element files for input and output.
  a text input file is generated based on keywords passed on to makegauinp.
  returns the names of the generated text input and output files.  the
  text output file has a generated name unless toutput is specified.
  progargs are any extra flags to pass to the command to run gaussian;
  the strings in the list do not require shell-type quoting, e.g.
  progargs=["-c=0-3","-m=2gb"] would set the CPUs and memory.

Module data
-----------

scalar_names  -- dict containing the scalar result values held in in an
                 internal array

mat_names_arr -- array containing the names of standard operator matrices,
                 which if written to a file are in a canonical order.

mat_names     -- dict containing the standard operator names, with values
                 giving the canonical order.

head_scalars  -- dict listing the scalars which are stored in header
                 records in the file.

head_arrays   -- dict listing the arrays which are setored in header
                 records in the file.

"""

__version__ = 2.0
import sys
import io
import re
import os
import subprocess
import tempfile
import numpy as np
INTSIZE_NAME = "GAUOPEN_INTSIZE"
doi8 = False
import qcmatrixio as qcmio
INTTYPE = "int32"
import QCOpMat as qco

# name of scalars in the /Gen/ areray
scalar_names_arr = ["VIRIAL","X-EFIELD","Y-EFIELD","Z-EFIELD","TE SCF ENERGY","SCRF G-FACTOR","SCRF A0",
  "THERMAL ENERGY","ECC","ECC(T)","EVAR1","ZPE","COMPOUND ENERGY","NIMAG","DEPUHF","EPUHF","ECBS2","ECBSI",
  "EPMP2","EPMP3","GEOM RMSF","ECIS-MP2","SCF RMSDP","S2-A","ECIS","DEUMP4D","EBDREF","EMP5","S4SD","EFC",
  "SCFTAU","ESCF","EUMP2","EUMP3","EUMP4","CBS OIII","EPRF","EMP4DQ","EMP4SDQ","SCALAR40","ENUCREP",
  "PSCFT","ETOTAL","S2SCF","S2-1","S2-D","A0","SCALAR48","TEMPERATURE","PRESSURE","FREQSCALE","INACTNUCREP",
  "DE2-SINGLES","DE2","RF NUC","RF ELEC","CURVATURE","IRC RC","EXTFLAG","ESCF1IT","JOB STATUS","NDERIV"]

# label for the scalars element in the file and in the object
LENGS = 1000
GSNAME = "GAUSSIAN SCALARS"

WLENBUF = 4000
WLENBFS = 2000
FRAGNAME = "INTEGER FRAGMENT"

# names for standard order of matrices in file, for writing

mat_names_arr = ["SHELL TO ATOM MAP","SHELL TYPES","NUMBER OF PRIMITIVES PER SHELL",
  "PRIMITIVE EXPONENTS","CONTRACTION COEFFICIENTS","P(S=P) CONTRACTION COEFFICIENTS",
  "COORDINATES OF EACH SHELL","BONDS PER ATOM","BONDED ATOMS","BOND TYPES","GAUSSIAN SCALARS",
  "INTEGER ISO","INTEGER SPIN","REAL ZEFFECTIVE","REAL GFACTOR","REAL ZNUCLEAR",
  "NUCLEAR GRADIENT","NUCLEAR FORCE CONSTANTS","ELECTRIC DIPOLE MOMENT",
  "ELECTRIC DIPOLE POLARIZABILITY","ELECTRIC DIPOLE DERIVATIVES",
  "DIPOLE POLARIZABILITY DERIVATIVES","ELECTRIC DIPOLE HYPERPOLARIZABILITY",
  "OVERLAP","CORE HAMILTONIAN ALPHA","CORE HAMILTONIAN BETA","KINETIC ENERGY",
  "ORTHOGONAL BASIS","DIPOLE INTEGRALS","QUADRUPOLE INTEGRALS",
  "OCTOPOLE INTEGRALS","HEXADECAPOLE INTEGRALS","DIP VEL INTEGRALS","R X DEL INTEGRALS",
  "ALPHA ORBITAL ENERGIES","BETA ORBITAL ENERGIES","ALPHA MO COEFFICIENTS",
  "BETA MO COEFFICIENTS","ALPHA DENSITY MATRIX","BETA DENSITY MATRIX",
  "ALPHA SCF DENSITY MATRIX","BETA SCF DENSITY MATRIX","ALPHA FOCK MATRIX",
  "BETA FOCK MATRIX","OVERLAP DERIVATIVES","CORE HAMILTONIAN DERIVATIVES",
  "F(X)","DENSITY DERIVATIVES","FOCK DERIVATIVES","ALPHA UX","BETA UX",
  "ALPHA MO DERIVATIVES","BETA MO DERIVATIVES","ALPHA SCF DENSITY","BETA SCF DENSITY",
  "ALPHA MP FIRST ORDER DENSITY","BETA MP FIRST ORDER DENSITY",
  "ALPHA MP2 DENSITY","BETA MP2 DENSITY","ALPHA MP3 DENSITY","BETA MP3 DENSITY",
  "ALPHA MP4 DENSITY","BETA MP4 DENSITY","ALPHA CI ONE-PARTICLE DENSITY",
  "BETA CI ONE-PARTICLE DENSITY","ALPHA CI DENSITY","BETA CI DENSITY",
  "ALPHA QCI/CC DENSITY","BETA QCI/CC DENSITY","ALPHA DENSITY CORRECT TO SECOND ORDER",
  "BETA DENSITY CORRECT TO SECOND ORDER","ALPHA ONIOM DENSITY","BETA ONIOM DENSITY",
  "GIAO D2H/DBDM","GIAO L/R3", "REGULAR 2E INTEGRALS", "RAFFENETTI 2E INTEGRALS",
  "AA MO 2E INTEGRALS", "AB MO 2E INTEGRALS", "BA MO 2E INTEGRALS", "BB MO 2E INTEGRALS"]

# scalars in the header records
head_scalars_arr = ["title","natoms","nbasis","nbsuse","icharg","multip","ne","iopcl",
  "icgu","nfc","nfv","itran","idum9","nshellao","nprimao","nshelldb","nprimdb","nbondtot"]
		       
# arrays in the header records
head_arrays_arr = ["ian","iattyp","atmchg","c","ibf","ibftyp","atmwgt"]

scalar_names = {name.upper():i for i,name in enumerate(scalar_names_arr)}
scalar_synonyms = {re.sub("[ ()-]*","",name).upper():name for name in scalar_names_arr}
mat_names = {name.upper():i for i,name in enumerate(mat_names_arr)}
mat_names_synonyms = {re.sub("[ ()-]*","",name).upper():name for name in mat_names_arr}
head_scalars = {name:i for i,name in enumerate(head_scalars_arr)}
head_arrays = {name:i for i,name in enumerate(head_arrays_arr)}

# default atomic weights, elements 0 to 109
defatw = [ 0.00000000,  1.00782504,  4.00260325,  7.01600450,  9.01218250,
          11.00930530, 12.00000000, 14.00307401, 15.99491464, 18.99840325,
          19.99243910, 22.98976970, 23.98504500, 26.98154130, 27.97692840,
          30.97376340, 31.97207180, 34.96885273, 39.96238310, 38.96370790,
          39.96259070, 44.95591360, 47.94794670, 50.94396250, 51.94050970,
          54.93804630, 55.93493930, 58.93319780, 57.93534710, 62.92959920,
          63.92914540, 68.92558090, 73.92117880, 74.92159550, 79.91652050,
          78.91833610, 83.91150640, 84.91170000, 87.90560000, 88.90540000,
          89.90430000, 92.90600000, 97.90550000, 98.90630000,101.90370000,
         102.90480000,105.90320000,106.90509000,113.90360000,114.90410000,
         117.90180000,120.90380000,129.90670000,126.90040000,131.90420000,
         132.90542900,137.90500000,138.90610000,139.90530000,140.90740000,
         141.90750000,144.91270000,151.91950000,152.92090000,157.92410000,
         158.92500000,163.92880000,164.93030000,165.93040000,168.93440000,
         173.93900000,174.94090000,179.94680000,180.94800000,183.95100000,
         186.95600000,189.95860000,192.96330000,194.96480000,196.96660000,
         201.97060000,204.97450000,207.97660000,208.98040000,208.98250000,
         210.98750000,222.01750000,223.01980000,226.02540000,227.02780000,
         232.03820000,231.03590000,238.05080000,237.04800000,242.05870000,
         243.06140000,246.06740000,247.07020000,249.07480000,252.08290000,
         252.08270000,255.09060000,259.10100000,262.10970000,261.10870000,
         262.11410000,266.12190000,264.12470000,  0.00000000,268.13880000]

maxan = len(defatw) - 1

def DimTuple (n1,n2,n3,n4,n5):
  if n5 > 1: return (n1,n2,n3,n4,n5)
  elif n4 > 1: return (n1,n2,n3,n4)
  elif n3 > 1: return (n1,n2,n3)
  elif n2 > 1: return (n1,n2)
  else: return((n1,))

def optfile (namei,suffix="",retfd=True):
  if namei is None:
    fi = tempfile.NamedTemporaryFile (mode='w+t',suffix=suffix,delete=False)
  else:
    fi = open (namei,"w+t")
  if retfd: return (fi)
  else:
    name = fi.name
    fi.close()
    return (name)

def makegauinp (matfi, matfo, tinput=None, dofock=False, motran=None,
                aotype=None, window=None, miscroute=None, model="HF",
                symm="nosymm", haveorbs=True, basis="ChkBasis",
                program="g16", revision="b01", progargs=[]):
# This routine operates in two ways, because g16a03 requires building
# an input file while for g16b01 and later and for gdv everything can
# be done using command-line arguments.  For the first case, the name of
# the input file is returned along with an unaltered copy of progargs;
# for the second case, None is returned instead of a file name and an
# updated progargs is returned with the appropriate switches.
  newpa = progargs;
  if revision is "a03":
    fi = optfile (tinput,suffix=".gjf")
    fi.write ("%oldmat=i4labels,")
    fi.write (matfi+"\n")
  else:
    fi = io.StringIO()
    if doi8: newpa.append ("-IM="+matfi)
    else: newpa.append ("-IM4="+matfi)
  fi.write ("#p "+model+" geom=allcheck " + basis + " test output=(matrix")
  if not doi8: fi.write (",i4labels")
  if motran is not None: fi.write (",mo2el")
  fi.write (") ")
  if symm != "": fi.write(symm+" ")
  if dofock is False:
    if haveorbs: fi.write("guess=(copychk,only)")
    else: fi.write("guess=(*none*,only)")
  elif dofock is True: fi.write ("guess=read scf=(novaracc,maxcyc=-1)")
  elif dofock.upper() == "DENSITY": fi.write ("guess=copychk scf=(novaracc,maxcyc=-1)")
  elif dofock.upper() == "SCF": fi.write ("scf=(novaracc)")
  elif dofock.upper() == "SCFREAD": fi.write ("guess=read scf=(novaracc)")
  else: raise TypeError
  if aotype is not None:
    fi.write(" scf=conventional ")
    if aotype == 0 or aotype == "regular" or aotype == "noraff": fi.write ("noraff")
    else: fi.write ("int=raf%d" % aotype)
  if motran == "partial": fi.write(" tran=iabc")
  elif motran == "full": fi.write(" tran=full")
  if window is not None: fi.write (" window="+str(window))
  if miscroute is not None: fi.write (" "+miscroute)
  if revision is "a03":
    fi.write("\n\n"+matfo+"\n\n")
    itemp = fi.name
  else:
    if doi8: newpa.append ("-OM="+matfo)
    else: newpa.append ("-OM4="+matfo)
    newpa.append ("-X="+fi.getvalue())
    itemp = None
  fi.close()
  return (itemp,newpa)

def rungau (matfi, matfo, program="g16", progargs=[], debug=False, toutput=None, **kwargs):
  itemp,pargs = makegauinp (matfi,matfo,program=program,progargs=progargs,**kwargs)
  otemp = optfile (toutput,suffix=".log",retfd=False)
  try:
    unlink (matfo)
  except:
    pass
  if itemp is None: fi = subprocess.DEVNULL
  else: fi = open (itemp,mode="r")
  fo = open (otemp,mode="w")
  pargs.insert (0,program)
  if debug: print ("rungau program",program,"progargs",pargs,"itemp",
                   itemp,"otemp",otemp,file=sys.stderr)
  subprocess.call(pargs,stdin=fi,stdout=fo)
  if itemp is not None: fi.close()
  fo.close()
  return (itemp,otemp)

class MatEl (object):

  def __init__ (self,debug=False,file=None,**kwargs):
    self.__DEBUG = debug
    self.__FH = None
    self.labfil = "Gaussian matrix elements"
    self.fversion = 2
    self.__NLAB = 11
    self.gversion = "QCMatEl.py %f" % 1.0
    self.title = "No title"
    self.__LENREC = 4000
    self.__LEN12L = 4
    self.__LEN4L = 4
    self.__REC11 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=INTTYPE)
    self.__GSCAL = np.zeros((LENGS))
    self.natoms = 0
    self.nbasis = 1
    self.nbsuse = 1
    self.icharg = 0
    self.multip = 0
    self.ne = 0
    self.iopcl = 0
    self.icgu = 0
    self.nfc = 0
    self.nfv = 0
    self.itran = 0
    self.idum9 = 0
    self.nshlao = 0
    self.nprmao = 0
    self.nshldb = 0
    self.nprmdb = 0
    self.nbondtot = 0
    self.ian = None
    self.iattyp = None
    self.atmchg = None
    self.atmwgt = None
    self.c = None
    self.ibfatm = np.array([0],dtype=INTTYPE)
    self.ibftyp = np.array([0],dtype=INTTYPE)
    self.__MATLIST = {}
    if file is not None:  self.read(file,**kwargs)
    
  @property
  def unit (self):  return self.__FH

  @unit.setter
  def unit (self,iu):
    self.__FH = iu
    return (self.__FH)

  @property
  def debug (self):  return self.__DEBUG

  @debug.setter
  def debug (self,value):
    self.__DEBUG = value
    return (self.__DEBUG)

  @property
  def matlist (self):  return self.__MATLIST

  @property
  def scalars (self):  return self.__GSCAL

  @property
  def nfrag (self):
    if FRAGNAME in self.__MATLIST: return max (self.__MATLIST[FRAGNAME].array)
    else:  return 0;

  def addobj (self,obj):
    name = obj.name.upper()
    if name in mat_names_synonyms: name = mat_names_synonyms[name]
    obj.name = name
    self.__MATLIST[obj.name] = obj

  def delobj (self,namei):
    name = namei.upper()
    if name in mat_names_synonyms: name = mat_names_synonyms[name]
    if name in self.__MATLIST: del self.__MATLIST[name]

  def scalar (self,namei,*val):
    name = namei.upper()
    if name in scalar_synonyms: name = scalar_synonyms[name]
    assert name in scalar_names
    if len(val) > 0: self.__GSCAL[scalar_names[name]] = val[0]
    return (self.__GSCAL[scalar_names[name]])

  def set_scalars (self,**kwargs):
    for name in kwargs: self.scalar (name,kwargs[name])

  def read (self,fname,check_status=False):
    if self.__DEBUG: print ("read file",fname)
    self.unit,self.labfil,self.fversion,self.__NLAB,self.gversion,self.title, \
      self.natoms,self.nbasis,self.nbsuse,self.icharg,self.multip,self.ne, \
      self.__LEN12L,self.__LEN4L,self.iopcl,self.icgu = qcmio.open_read (fname)
    if self.unit < 1:
      print ("failed to open matrix element file",fname," for reading.")
      raise IOError
    self.labfil = self.labfil.rstrip().decode("utf-8")
    self.gversion = self.gversion.rstrip().decode("utf-8")
    self.title = self.title.rstrip().decode("utf-8")
    self.ian,self.iattyp,self.atmchg,self.c,self.ibfatm,self.ibftyp,self.atmwgt, \
      self.nfc,self.nfv,self.itran,self.idum9,self.nshlao,self.nprmao,\
      self.nshldb,self.nprmdb,self.nbondtot = \
      qcmio.rd_head (self.unit,self.__NLAB,self.natoms,self.nbasis)
    gotone = True
    while (gotone):
      cbuf,ni,nr,ntot,lenbuf,n1,n2,n3,n4,n5,asym,nri,eof = qcmio.rd_labl(self.unit,self.fversion)
      cbuf = cbuf.rstrip().decode("utf-8")
      gotone = not eof
      if nri == 2: type = "c"
      else: type = "d"
      if (gotone): 
        dimens = DimTuple (n1,n2,n3,n4,n5)
        lr = qcmio.lenarr(n1,n2,n3,n4,n5)
        if (ni >= 1) and (nr == 0):
          arr = qcmio.rd_ibuf(self.unit,ni*ntot,ni*lenbuf)
          myobj = qco.OpMat (cbuf,arr,asym=asym,nelem=ni,dimens=dimens)
        elif (ni == 0) and (nr >= 1):
          if (nri == 1):  arr = qcmio.rd_rbuf(self.unit,nr*ntot,nr*lenbuf)
          else: arr = qcmio.rd_cbuf(self.unit,nr*ntot,nr*lenbuf)
          myobj = qco.OpMat (cbuf,arr,asym=asym,nelem=nr,dimens=dimens)
        elif qcmio.aoints(cbuf):
          if nr == 1: arr = qcmio.rd_2e1 (self.unit,lr,ntot,lenbuf)
          else:
            arr = qcmio.rd_2en (self.unit,nr,lr,lr*nr,ntot,lenbuf)
            arr = arr.reshape((lr,nr),order='F')
          myobj = qco.OpMat (cbuf,arr,nelem=nr,dimens=dimens)
        elif (ni == 1):
          lnz,arr = qcmio.rd_rind(self.unit,nr,lr,ntot,lenbuf)
          if nr == 1: arr = np.reshape(arr,(lr),order='F')
          else: arr = arr.T
          myobj = qco.OpMat (cbuf,arr,asym=asym,nelem=nr,dimens=dimens)
        else:
          raise IOError
          qcmio.rd_skip (self.unit,ntot,lenbuf)
        if cbuf == GSNAME: self.__GSCAL = arr
        else: self.__MATLIST[cbuf] = myobj
    qcmio.close_matf (self.unit)
    if check_status: assert self.scalar("JOB STATUS") == 1.0

  def print(self,doinp=False,**kwargs):
    if doinp:
      sep = "\n"
      f2d = "%d"
      f3d = "%d"
      f6d = "%d"
      f8d = "%d"
    else:
      sep = " "
      f2d = "%2d"
      f3d = "%3d"
      f6d = "%6d"
      f8d = "%8d"
    fstr = "Label=%s" + sep + "IVers=" + f2d + sep + "NLab=" + f2d + sep + "Version=%s"
    print (fstr % (self.labfil,self.fversion,self.__NLAB,self.gversion),**kwargs)
    print ("Title=%s" % self.title,**kwargs)
    fstr = "NAtoms=" + f6d + sep + "NBasis=" + f6d + sep + "NBsUse=" + f6d + sep + "ICharg=" + f6d + \
      sep + "Multip=" + f6d + sep + "NE=" + f6d + sep + "Len12L=%1d" + sep + "Len4L=%1d" + \
      sep + "IOpCl=" + f6d + sep + "ICGU=" + f3d
    print (fstr % (self.natoms,self.nbasis,self.nbsuse,self.icharg,self.multip,self.ne,
                   self.__LEN12L,self.__LEN4L,self.iopcl,self.icgu),**kwargs)
    qco.print1d (False,"i",1,"IAn",self.ian,doinp=doinp,**kwargs)
    qco.print1d (False,"i",1,"IAtTyp",self.iattyp,doinp=doinp,**kwargs)
    qco.print1d (False,"d",1,"AtmChg",self.atmchg,doinp=doinp,**kwargs)
    qco.print1d (False,"d",2,"C",self.c,doinp=doinp,**kwargs)
    qco.print1d (False,"i",2,"IBfAtm",self.ibfatm,doinp=doinp,**kwargs)
    qco.print1d (False,"i",2,"IBfTyp",self.ibftyp,doinp=doinp,**kwargs)
    qco.print1d (False,"d",1,"AtmWgt",self.atmwgt,doinp=doinp,**kwargs)
    fstr = "NFC=" + f6d + sep + "NFV=" + f6d + sep + "ITran=" + f6d
    print (fstr % (self.nfc,self.nfv,self.itran),**kwargs)
    fstr = "NShlAO=" + f8d + sep + "NPrmAO=" + f8d + sep + "NShlDB=" + \
      f8d + sep + "NPrmDB=" + f8d + sep + "NBTot=" + f8d
    print (fstr % (self.nshlao,self.nprmao,self.nshldb,self.nprmdb,self.nbondtot),**kwargs)
    for lab in mat_names_arr:
      if lab == GSNAME:
        if not qco.doinpprt (GSNAME,self.__GSCAL,doinp=doinp,**kwargs):
          qco.printlab (GSNAME,0,1,1,len(self.__GSCAL),0,len(self.__GSCAL),1,1,1,1,0,doinp=doinp,**kwargs)
          qco.print1d (True,"d",5," ",self.__GSCAL,doinp=doinp,**kwargs)
      else:
        if lab in self.__MATLIST:
          if not qco.doinpprt (lab,self.__MATLIST[lab].array,doinp=doinp,**kwargs):
            self.__MATLIST[lab].print_mat(doinp=doinp,**kwargs)
    for lab in sorted(self.__MATLIST):
      if not lab in mat_names:
        if not qco.doinpprt (lab,self.__MATLIST[lab].array,doinp=doinp,**kwargs):
          self.__MATLIST[lab].print_mat(**kwargs)

  def __str__ (self):
    stream = io.StringIO()
    self.print(file=stream)
    str = stream.getvalue()
    return (str[:-1])

  def write (self,fname):
    if self.__DEBUG: print ("write file",fname)
    self.unit = qcmio.open_write (fname,self.labfil,self.gversion,self.title,
      self.natoms,self.nbasis,self.nbsuse,self.icharg,self.multip,
      self.ne,self.iopcl,self.icgu)
    if self.unit < 1:
      print ("failed to open matrix element file",fname," for writing.")
      raise IOError
    qcmio.wr_head (self.unit,self.ian,self.iattyp,
      self.atmchg,self.c,self.ibfatm,self.ibftyp,self.atmwgt,self.nfc,
      self.nfv,self.itran,self.idum9,self.nshlao,self.nprmao,self.nshldb,
      self.nprmdb,self.nbondtot)
    for lab in mat_names_arr:
      if lab == GSNAME:
        y = self.__GSCAL.reshape((1,self.__GSCAL.size),order='F')
        nnz = max(qcmio.numnzr(y),1)
        qcmio.wr_labl(self.unit,GSNAME,1,1,nnz,WLENBFS,LENGS,1,1,1,1,0)
        qcmio.wr_rind(self.unit,nnz,WLENBFS,y)
      else:
        if lab in self.__MATLIST: self.__MATLIST[lab].write(self.unit,WLENBUF)
    for lab in sorted(self.__MATLIST):
      if not lab in mat_names: self.__MATLIST[lab].write(self.unit,WLENBUF)
    qcmio.wr_labl(self.unit,"END",0,0,0,0,0,0,0,0,0,False)
    qcmio.close_matf(self.unit)

  def update (self, matfi=None, matfo=None, check_status=True, doinit=False, **kwargs):
    matinp = optfile (matfi,suffix=".mat",retfd=False)
    matout = optfile (matfo,suffix=".mat",retfd=False)
    self.write (matinp)
    itemp,otemp = rungau (matinp,matout,**kwargs)
    if doinit:  self.__init__()
    try:
      self.read (matout,check_status=check_status)
    except:
      print ("Gaussian failed; retaining matinp",matinp,"matout",matout,
             "input",itemp,"output",otemp,file=sys.stderr)
      return (False,itemp,otemp)
    if itemp is not None and ("tinput" not in kwargs or kwargs["tinput"] is None):
      os.unlink (itemp)
    if "toutput" not in kwargs or kwargs["toutput"] is None: os.unlink (otemp)
    if matfi is None: os.unlink (matinp)
    if matfo is None: os.unlink (matout)
    return (True,None,None)

  def load_head (self,title="No title",natoms=None,nbasis=None,nbsuse=None,
                 icharg=0,multip=1,ne=None,iopcl=0,icgu=None,nfc=0,nfv=0,
                 itran=0,ian=None,iattyp=None,atmchg=None,atznuc=None,
                 c=None,atmwgt=None,labfil="Gaussian matrix elements",
                 progversion="QCMatEl.py 1.0"):
    self.title = title
    self.labfil = labfil
    self.gversion = progversion
    assert natoms is not None
    self.natoms = natoms
    if nbasis is None: self.nbasis = 1
    else: self.nbasis = nbasis
    if nbsuse is None: self.nbsuse = self.nbasis
    else:  self.nbsuse = nbsuse
    self.icharg = icharg
    self.multip = multip
    self.ne = ne
    self.iopcl = iopcl
    if icgu is None:
      if iopcl == 6: self.icgu = 221
      else: self.icgu = 10*((iopcl % 4)//2) + (iopcl % 2) + 111
    else: self.icgu = icgu
    self.nfc = nfc
    self.nfv = nfv
    self.itran = itran
    self.idum9 = 0
    self.nshlao = 0
    self.nprmao = 0
    self.nshldb = 0
    self.nprmdb = 0
    self.nbondtot = 0
    assert len(ian) == self.natoms
    self.ian = np.array(ian,dtype=INTTYPE)
    if iattyp is None: self.iattyp = np.zeros((natoms,),dtype=INTTYPE)
    else:
      assert len(iattyp) == self.natoms
      self.iattyp = np.array(iattyp,dtype=INTTYPE)
    if atmchg is None: self.atmchg = np.array(self.ian,dtype="float64")
    else:
      assert len(atmchg) == self.natoms
      self.atmchg = np.array(atmchg,dtype="float64")
    assert len(c) == (3*self.natoms)
    self.c = np.array(c,dtype="float64")
    if atmwgt is None:
      self.atmwgt = np.array([defatw[min(max(ia,0),maxan)] for ia in self.ian],dtype="float64")
    else:
      assert len(atmwgt) == self.natoms
      self.atmwgt = np.array(atmwgt,dtype="float64")
    self.ibfatm = np.zeros((self.nbasis,),dtype=INTTYPE)
    self.ibftyp = np.zeros((self.nbasis,),dtype=INTTYPE)
    self.__MATLIST = {}
    for i in range(len(self.__GSCAL)): self.__GSCAL[i] = 0.0e0
    if atznuc is None: znuc = self.atmchg
    else:
      assert len(atznuc) == self.natoms
      znuc = np.array(atznuc,dtype="float64")
    self.addobj(qco.OpMat("REAL ZNUCLEAR",znuc))
