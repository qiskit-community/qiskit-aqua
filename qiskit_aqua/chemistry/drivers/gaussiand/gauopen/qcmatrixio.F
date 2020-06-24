*Deck Open_Read
      Subroutine Open_Read(Name,IU,LabFil,IVers,NLab,GVers,Title,NAtoms,
     $  NBasis,NBsUse,ICharg,Multip,NE,Len12L,Len4L,IOpCl,ICGU)
      Implicit None
C
C     This file contains low-level routines to read and write matrix
C     element files as Fortran unformatted file.  All routines can
C     be called from Fortran programs and are wrapped for use in
C     Python programs, usually via the QCMatEl and QCOpMat
C     modules/classes.  The file format and contents of the header
C     records are documented in unfdat.txt.
C
C     Routines defined here, with Fortran and Python call sequences:
C
C     Call Open_Read(Name,IU,LabFil,IVers,NLab,GVers,Title,NAtoms,
C    $  NBasis,NBsUse,ICharg,Multip,NE,Len12L,Len4L,IOpCl,ICGU)
C     (iu,labfil,ivers,nlab,gvers,title,natoms,nbasis,nbsuse,icharg,
C       multip,ne,len12l,len4l,iopcl,icgu) = open_read(name)
C     Open the named matrix-element file for reading, and return
C     the listed scalars from the initial 2 header records.  IU receives
C     the Fortran unit number of the open file, or -1 if the open failed.
C
C     Call Open_Write(Name,IU,LabFil,GVers,Title,NAtoms,NBasis,
C    $  NBsUse,ICharg,Multip,NE,IOpCl,ICGU)
C     iu = open_write(name,labfil,gvers,title,natoms,nbasis,nbsuse,
C       icharg,multip,ne,iopcl,icgu)
C     Open the named matrix-element file for writing and write the
C     named scalars to the initial 2 header records.  IU receives the
C     Fortran unit number of the open file, or -1 if the open failed.
C
C     Call Close_MatF(IU)
C     close_matf (iu)
C     Close the file open on Fortran unit IU.
C
C     Ind = Lind2C(Check,N1,N2,ASym,I,J,Sign)
C     Ind = Lind2(Check,N1,N2,ASym,I,J,Sign)
C     Ind = Lind3C(Check,N1,N2,N3,ASym,I,J,K,Sign)
C     Ind = Lind3(Check,N1,N2,N3,ASym,I,J,K,Sign)
C     Ind = Lind4C(Check,N1,N2,N3,N4,ASym,I,J,K,L,Sign)
C     Ind = Lind4(Check,N1,N2,N3,N4,ASym,I,J,K,L,Sign)
C     Ind = Lind5C(Check,N1,N2,N3,N4,N5,ASym,I,J,K,L,M,Sign)
C     Ind = Lind5(Check,N1,N2,N3,N4,N5,ASym,I,J,K,L,M,Sign)
C     (ind,sign) = lind2c(check,n1,n2,ASym,i,j), etc.
C
C     Return the 0-based index into a linear array given 2, 3, 4, or 5
C     indices and dimensions.  Check is True to check for indices in
C     range and return -1 if they are out of range; otherwise the
C     indices are assumed to be valid.  The C versions take c-style
C     indices (0-based and rightmost dimension and index fastest
C     running) while the others are Fortran-style (1-based and
C     leftmost dimension and index fastest running).  Sign is +/-1
C     to indicate whether the upper or lower triangle was selected
C     (i.e., whether to apply a sign flip for anti-symmetric/Hermetian
C     matrices and/or take a complex conjugate.
C     Note the all functions return a 0-based index.
C
C     Call Rd_Head(IU,NLab,NAtoms,NBasis,IAn,IAtTyp,AtmChg,C,
C    $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,NShlDB,
C    $  NPrmDB,NBTot)
C     (ian,iattyp,atmchg,c,ibfatm,ibftyp,atmwgt,nfc,nfv,itran,idum9,
C       nshlao,nprmao,nshldb,nprmdb,nbtot) = rd_head(iu,nlab,natoms,nbasis)
C     Read the remaining header records (3 to NLab) and return the named
C     arrays.
C
C     Call Rd_Labl(IU,IVers,CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,ASym,
C    $  NRI,EOF)
C     (cbuf,ni,nr,ntot,lenbuf,n1,n2,n3,n4,n5,ASym,nri,eof) = rd_labl(iu,ivers)
C     read the label record for an operator matrix.
C
C     Call Rd_IBuf(IU,NI*NTot,NR*LenBuf,Arr)
C     arr = rd_ibuf(iu,ni*ntot,ni*lenbuf)
C     Call Rd_RBuf(IU,NR*NTot,NR*LenBuf,Arr)
C     arr = rd_rbuf(iu,nr*ntot,nr*lenbuf)
C     Call Rd_CBuf(IU,NR*NTot,NR*LenBuf,Arr)
C     arr = rd_cbuf(iu,nr*ntot,nr*lenbuf)
C     read an integer/real/complex array from the file given parameters
C     from the header record.
C
C     Call Rd_RInd(IU,NR,LR,NTot,LenBuf,LNZ,RArr)
C     lnz,arr = rd_rind(iu,nr,lr,ntot,lenbuf)
C     Read a real array stored with indices for non-zero elements.
C     lnz receives the index (1-based) of the last non-zero element.
C
C     Call Rd_Skip(IU,NTot,LenBuf)
C     rd_skip(iu,ntot,lenbuf)
C     Skip the data records for an object on the file having NTot
C     elements stored with LenBuf per record.
C
C     Call Rd_2E1(IU,LR,NTot,LenBuf,RArr)
C     Call Rd_2EN(IU,NR,LR,LRNR,NTot,LenBuf,RArr)
C     arr = rd_2e1(iu,lr,ntot,lenbuf)
C     arr = rd_2en(iu,nr,lr,lrnr,ntot,lenbuf)
C     Read and return an array of AO 2e integrals (really, a
C     4-dimensional array with quartets of indices and one
C     value or NR values per index set).  NTot is the number of non-zero
C     values (from the header record for the object) and
C     LR is the total number of elements (LenArr(-N,-N,-N,N,1), where
C     N is the number of basis functions).
C
C     Call Wr_Head(IU,NAtoms,NAt3,NBasis,IAn,IAtTyp,AtmChg,C,
C    $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,NShlDB,
C    $  NPrmDB,NBTot)
C     wr_head(iu,natoms,nat3,nbasis,ian,iattyp,atmchg,c,ibfatm,ibftyp,
C       atmwgt,nfc,nfv,itran,idum9,nshlao,nprmao,nshldb,nprmdb,nbtot)
C     Write the header records (3 to NLab) to Fortran unit IU.
C
C     The following low-level routines are not usually called directly,
C     but are accessed via the routine Wr_L{IBuf,RBuf,CBuf,RInd} which
C     are in qcmatrix.F for fortran and QCMatEl.py for Python.
C
C     Call Wr_Labl(IU,CBuf,NI,NR,NTot,LenBfX,N1,N2,N3,N4,N5,ASym)
C     Write the header record for one matrix to the file.
C
C     Call Wr_IBuf(IU,NTot,LenBfX,Arr)
C     Call Wr_RBuf(IU,NTot,LenBfX,Arr)
C     Call Wr_CBuf(IU,NTot,LenBfX,Arr)
C     Call Wr_RInd(IU,NR,LR,NTot,LenBfX,RArr)
C     Call Wr_2E(IU,NTot,NR,N,LR,LenBfX,RArr)
C     Write objects of the specified types to the file.  LenBfX is
C     derived from the general LenBuf based on the length of the
C     items; this is normally handled by the Wr_Lxxxx routines.
C
C     LVal = AOInts(CBuf)
C     lval = aoints(cbuf)
C     Return true if the operator identified by the string in CBuf
C     is an AO 2e integral array (regular or Rafenetti).
C
C     NTot = LenArr(N1,N2,N3,N4,N5)
C     ntot = lenarr(n1,n2,n3,n4,n5)
C     return the total number of index values of an array with
C     the specified dimensions, accounting for possible
C     lower-triangular indices.  This does not include a
C     possible multiple number of values per index (NI or NR in
C     the record header, nelem in the object).
C
C     NNZ = NumNZA(NR,NTot,X)
C     nnz = numnza(x)
C     return the number of non-zero elements of X(NTot,NR) (Fortran order).
C
C     NNZ = NumNZR(NR,NTot,X)
C     nnz = numnzr(x)
C     return the number of non-zero elements of X(NR,NTot) (Fortran order).
C
C     Open a Gaussian Matrix-element file and read header information.
C     IU receives the Fortran unit number or -1 if the open failed.
C
      Integer LStr, IUUse, Len12D, Len4D, IUSt, IUEnd
      Parameter (Len12D=4,Len4D=4)
      Parameter (LStr=64,IUSt=57,IUEnd=99)
C     The latest f2py is now "improved" and is now too stupid to handle
C     character string lengths given by parameters.
C     Character*(*) Name, LabFil*(LStr), GVers*(LStr), Title*(LStr)
      Character*(*) Name, LabFil*64, GVers*64, Title*64
      Logical IsOpen
      Integer IU,IVers,NLab,NAtoms,NBasis,NBsUse,ICharg,Multip,NE,
     $  Len12L,Len4L,IOpCl,ICGU
cf2py intent(out) labfil,gvers,title,iu,ivers,nlab,natoms,nbasis,nbsuse
cf2py intent(out) icharg,multip,ne,len12l,len4l,iopcl,icgu
 1000 Format(' This QCMatrixIO was compiled with Len12=',I1,' Len4=',I1,
     $  ' but file has Len12=',I1,' Len4=',I1,'.')
C
      LabFil = ' '
      IVers = 0
      NLab = 0
      GVers = ' '
      Title = ' '
      NAtoms = 0
      NBasis = 0
      NBsUse = 0
      ICharg = 0
      Multip = 0
      NE = 0
      Len12L = 0
      Len4L = 0
      IOpCl = 0
      ICGU = -1
      Do 10 IUUse = IUSt, IUEnd
        Inquire(Unit=IUUse,Opened=IsOpen)
        If(.not.IsOpen) goto 20
   10   Continue
      IU = -1
      Return
C
   20 Open (Unit=IUUse,File=Name,Form='Unformatted',Status='Old',
     $  Err=900)
      IU = IUUse
      Read(IU) LabFil(1:LStr), IVers, NLab, GVers(1:LStr)
      If(IVers.eq.1) then
        Read(IU) Title(1:LStr), NAtoms, NBasis, NBsUse, ICharg, Multip,
     $    NE, Len12L, Len4L
      else
        Read(IU) Title(1:LStr), NAtoms, NBasis, NBsUse, ICharg, Multip,
     $    NE, Len12L, Len4L, IOpCl, ICGU
        endIf
      If(Len4L.ne.Len4D.or.Len12L.ne.Len12D) then
        Write(6,1000) Len12D, Len4D, Len12L, Len4L
        Goto 900
        endIf
      Return
C
  900 IU = -1
      Return
      End
*Deck Open_Write
      Subroutine Open_Write(Name,IU,LabFil,GVers,Title,NAtoms,NBasis,
     $  NBsUse,ICharg,Multip,NE,IOpCl,ICGU)
      Implicit None
C
C     Open a Gaussian Matrix-element file and write header information.
C     IU receives the Fortran unit number or -1 if the open failed.
C
      Integer LStr,IUUse,IVers,Len12L,Len4L,NLab,IUSt,IUEnd
      Parameter (Len12L=4,Len4L=4)
      Parameter (LStr=64,IUSt=57,IUEnd=99,IVers=2,NLab=11)
C     The latest f2py is now "improved" and is now too stupid to handle
C     character string lengths given by parameters.
C     Character*(*) Name, LabFil, GVers, Title, LLabFil*(LStr),
C    $  LGVers*(LStr), LTitle*(LStr)
      Logical IsOpen
      Character*(*) Name, LabFil, GVers, Title, LLabFil*64,
     $  LGVers*64, LTitle*64
      Integer IU,NAtoms,NBasis,NBsUse,ICharg,Multip,NE,IOpCl,ICGU
CF2PY Intent(Out) IU
C
      Do 10 IUUse = IUSt, IUEnd
        Inquire(Unit=IUUse,Opened=IsOpen)
        If(.not.IsOpen) goto 20
   10   Continue
      IU = -1
      Return
C
   20 Open (Unit=IUUse,File=Name,Form='Unformatted',Status='Unknown',
     $  Err=900)
      IU = IUUse
      LLabFil = LabFil
      LGVers = GVers
      LTitle = Title
      Write(IU) LLabFil, IVers, NLab, LGVers
      Write(IU) LTitle, NAtoms, NBasis, NBsUse, ICharg, Multip, NE,
     $  Len12L, Len4L, IOpCl, ICGU
      Return
C
  900 IU = -1
      Return
      End
*Deck Close_MatF
      Subroutine Close_MatF(IU)
      Implicit None
C
C     Close a matrix-element file.
C
      Integer IU
C
      Close (Unit=IU)
      Return
      End
*Deck AOInts
      Logical Function AOInts(CBuf)
      Implicit None
      Character*(*) CBuf, Reg, Raf
      Parameter (Reg='REGULAR 2E INTEGRALS',
     $  Raf='RAFFENETTI 2E INTEGRALS')
C
      AOInts = CBuf.eq.Reg.or.CBuf.eq.Raf
      Return
      End
*Deck LenArr
      Integer Function LenArr(N1,N2,N3,N4,N5)
      Implicit None
      Integer N1,N2,N3,N4,N5,N1X,N2X,N3X,N4X,N5X,Abs,Lind5,Sign
C
      N1X = N1
      If(N1X.eq.0) N1X = 1
      N2X = N2
      If(N2X.eq.0) N2X = 1
      N3X = N3
      If(N3X.eq.0) N3X = 1
      N4X = N4
      If(N4X.eq.0) N4X = 1
      N5X = N5
      If(N5X.eq.0) N5X = 1
      LenArr = Lind5(.False.,N1X,N2X,N3X,N4X,N5X,.False.,Abs(N1X),
     $  Abs(N2X),Abs(N3X),Abs(N4X),Abs(N5X),Sign) + 1
      Return
      End
*Deck LInd2C
      Integer Function LInd2C(Check,N1,N2,ASym,I,J,Sign)
      Implicit None
C
C     Linear or square indexing, I,J are 0-based, c order
C     output is 0-based.  Sign is +/-1.
C
      Logical Check,ASym
      Integer N1,N2,I,J,Lind2,Sign
CF2PY Intent (Out) Sign
C
      Lind2C = Lind2(Check,N2,N1,ASym,J+1,I+1,Sign)
      Return
      End
*Deck LInd2
      Integer Function Lind2(Check,N1,N2,ASym,I,J,Sign)
      Implicit None
C
C     Linear or square indexing, I,J are 1-based,
C     output is 0-based.  Sign is +/-1.
C
      Logical Check,ASym
      Integer N1,N2,I,J,Sign
CF2PY Intent (Out) Sign
C
      Sign = 1
      If(Check.and.(N2.le.0.or.N1.eq.0.or.(N1.lt.0.and.N1.ne.(-N2)).or.
     $  I.lt.1.or.I.gt.Abs(N1).or.J.lt.1.or.J.gt.N2)) then
        Lind2 = -1
        Return
        endIf
      If(N1.lt.0) then
        If(I.ge.J) then
          Lind2 = (I*(I-1))/2 + J - 1
        else
          Lind2 = (J*(J-1))/2 + I - 1
          Sign = -1
          endIf
      else
        Lind2 = N1*(J-1) + I - 1
        endIf
      Return
      End
*Deck Lind3C
      Integer Function Lind3C(Check,N1,N2,N3,ASym,I,J,K,Sign)
      Implicit None
C
C     Linear or square indexing, I,J,K are 0-based, c order
C     output is 0-based.  Sign is +/-1.
C
      Logical Check,ASym
      Integer N1,N2,N3,I,J,K,Lind3,Sign
CF2PY Intent (Out) Sign
C
      Lind3C = Lind3(Check,N3,N2,N1,ASym,K+1,J+1,I+1,Sign)
      Return
      End
*Deck LInd3
      Integer Function Lind3(Check,N1,N2,N3,ASym,I,J,K,Sign)
      Implicit None
C
C     Linear or square indexing, I,J,K are 1-based,
C     output is 0-based.  Sign is +/-1.
C
      Logical Check,ASym
      Integer N1,N2,N3,I,J,K,I1,J1,K1,N12,IJ,LInd2,Sign
CF2PY Intent (Out) Sign
C
      Sign = 1
      If(Check.and.(N3.le.0.or.(N1*N2).eq.0.or.
     $  (N1.lt.0.and.N1.ne.(-Abs(N2))).or.
     $  (N2.lt.0.and.N2.ne.(-N3)).or.I.lt.1.or.I.gt.Abs(N1).or.
     $  J.lt.1.or.J.gt.Abs(N2).or.K.lt.1.or.K.gt.N3)) then
        Lind3 = -1
        Return
        endIf
      I1 = I - 1
      J1 = J - 1
      K1 = K - 1
      If(N1.gt.0) then
        If(N2.gt.0) then
          LInd3 = N1*(N2*K1+J1) + I1
        else
          Lind3 = N1*Lind2(.False.,N2,N3,ASym,J,K,Sign) + I1
          endIf
      else if(N2.gt.0) then
        N12 = (N2*(N2+1))/2
        If(I.ge.J) then
          IJ = (I*I1)/2 + J1
        else
          IJ = (J*J1)/2 + I1
          Sign = -1
          endIf
        Lind3 = N12*K1 + IJ
      else
        K1 = Max(I,J,K) - 1
        I1 = Min(I,J,K) - 1
        J1 = I + J + K - K1 - I1 - 3
        LInd3 = I1 + (J1*(J1+1))/2 + (K1*(K1+1)*(K1+2))/6
        endIf
      Return
      End
*Deck Lind4C
      Integer Function Lind4C(Check,N1,N2,N3,N4,ASym,I,J,K,L,Sign)
      Implicit None
C
C     Linear or square indexing, I,J,K,L are 0-based, c order
C     output is 0-based.  Sign is +/-1.
C
      Logical Check,ASym
      Integer N1,N2,N3,N4,I,J,K,L,Lind4,Sign
CF2PY Intent (Out) Sign
C
      Lind4C = Lind4(Check,N4,N3,N2,N1,ASym,L+1,K+1,J+1,I+1,Sign)
      Return
      End
*Deck LInd4
      Integer Function Lind4(Check,N1,N2,N3,N4,ASym,I,J,K,L,Sign)
      Implicit None
C
C     Linear or square indexing, I,J,K,L are 1-based,
C     output is 0-based.  Sign is +/-1.
C
      Logical Check,ASym
      Integer N1,N2,N3,N4,I,J,K,L,I1,J1,K1,L1,ICase,Lind2,N23,
     $  Lind3,N123,KL,IJ,IJK,JKL,JK,N12,Sign,SignIJ,SignKL
CF2PY Intent (Out) Sign
C
      If(Check.and.(N4.le.0.or.(N1*N2*N3).eq.0.or.
     $  (N1.lt.0.and.N1.ne.(-Abs(N2))).or.
     $  (N2.lt.0.and.N2.ne.(-Abs(N3))).or.
     $  (N3.lt.0.and.N3.ne.(-N4)).or.
     $  I.lt.1.or.I.gt.Abs(N1).or.J.lt.1.or.J.gt.Abs(N2).or.
     $  K.lt.1.or.K.gt.Abs(N3).or.L.lt.1.or.L.gt.N4)) then
        Lind4 = -1
        Sign = 1
        Return
        endIf
      I1 = I - 1
      J1 = J - 1
      K1 = K - 1
      L1 = L - 1
      ICase = 0
      If(N1.lt.0) ICase = ICase + 1
      If(N2.lt.0) ICase = ICase + 2
      If(N3.lt.0) ICase = ICase + 4
      Goto (100,110,120,130,140,150,160,170), ICase+1
C
C     No symmetries.
  100 Lind4 = N1*(N2*(N3*L1+K1)+J1) + I1
      Sign = 1
      Return
C
C     I<=J
  110 IJ = Lind2(.False.,N1,N2,ASym,I,J,Sign)
      N12 = (N2*(N2+1))/2
      Lind4 = N12*(N3*L1+K1) + IJ
      Return
C
C     I,J<=K,L
  120 JK = Lind2(.False.,N2,N3,ASym,J,K,Sign)
      N23 = (N3*(N3+1))/2
      Lind4 = N1*(N23*L1+JK) + I1
      Return
C
C     I<=J<=K,L
  130 IJK = Lind3(.False.,N1,N2,N3,ASym,I,J,K,Sign)
      N123 = (N3*(N3+1)*(N3+2))/6
      Lind4 = N123*L1 + IJK
      Return
C
C     I,J,K<=L
  140 KL = Lind2(.False.,N3,N4,ASym,K,L,Sign)
      Lind4 = N1*(N2*KL+J1) + I1
      Return
C
C     I<=J,K<=L
  150 IJ = Lind2(.False.,N1,N2,ASym,I,J,SignIJ)
      KL = Lind2(.False.,N3,N4,ASym,K,L,SignKL)
      N12 = (N2*(N2+1))/2
      Lind4 = N12*KL + IJ
      Sign = SignIJ*SignKL
      Return
C
C     I,J<=K<=L
  160 JKL = Lind3(.False.,N2,N3,N4,ASym,J,K,L,Sign)
      Lind4 = N1*JKL + I1
      Return
C
C     I<=J<=K<=L
  170 IJ = Lind2(.False.,N1,N4,ASym,I,J,SignIJ)
      KL = Lind2(.False.,N3,N4,ASym,K,L,SignKL)
      Lind4 = Lind2(.False.,N1,N4,ASym,IJ+1,KL+1,Sign)
      Sign = Sign*SignIJ*SignKL
      Return
      End
*Deck Lind5C
      Integer Function Lind5C(Check,N1,N2,N3,N4,N5,ASym,I,J,K,L,M,
     $  Sign)
      Implicit None
C
C     Linear or square indexing, I,J,K,L,M are 0-based, c order
C     output is 0-based.  Sign is +/-1.
C
      Logical Check,ASym
      Integer N1,N2,N3,N4,N5,I,J,K,L,M,Lind5,Sign
CF2PY Intent (Out) Sign
C
      Lind5C = Lind5(Check,N5,N4,N3,N2,N1,ASym,M+1,L+1,K+1,J+1,I+1,
     $  Sign)
      Return
      End
*Deck LInd5
      Integer Function Lind5(Check,N1,N2,N3,N4,N5,ASym,I,J,K,L,M,Sign)
      Implicit None
C
C     Linear or square indexing, I,J,K,L,M are 1-based, output is
C     0-based.  Sign is +/-1, LM indices can not be lower triangular.
C
      Logical Check,ASym
      Integer N1,N2,N3,N4,N5,I,J,K,L,M,M1,Lind4,N1A,N2A,N3A,N1234,Sign
CF2PY Intent (Out) Sign
C
      If(Check.and.(N5.le.0.or.N4.le.0.or.(N1*N2*N3).eq.0.or.
     $  (N1.lt.0.and.N1.ne.(-Abs(N2))).or.
     $  (N2.lt.0.and.N2.ne.(-Abs(N3))).or.
     $  (N3.lt.0.and.N2.ne.(-N4)).or.
     $  I.lt.1.or.I.gt.Abs(N1).or.J.lt.1.or.J.gt.Abs(N2).or.
     $  K.lt.1.or.K.gt.Abs(N3).or.L.lt.1.or.L.gt.N4)) then
        Lind5 = -1
        Sign = 1
        Return
        endIf
      M1 = M - 1
      N1A = Abs(N1)
      N2A = Abs(N2)
      N3A = Abs(N3)
      N1234 = Lind4(.False.,N1,N2,N3,N4,ASym,N1A,N2A,N3A,N4,Sign) + 1
      Lind5 = Lind4(.False.,N1,N2,N3,N4,ASym,I,J,K,L,Sign) + N1234*M1
      Return
      End
*Deck NumNZA
      Integer Function NumNZA(NR,NTot,X)
      Implicit None
      Integer NR,NTot,I,J,I1
      Real*8 X(NTot,NR),Zero
      Parameter (Zero=0.0d0)
C
      NumNZA = 0
      Do 20 I = 1, NTot
        I1 = 0
        Do 10 J = 1, NR
          If(X(I,J).ne.Zero) I1 = 1
   10     Continue
   20   NumNZA = NumNZA + I1
      Return
      End
*Deck NumNZR
      Integer Function NumNZR(NR,NTot,X)
      Implicit None
      Integer NR,NTot,I,J,I1
      Real*8 X(NR,NTot),Zero
      Parameter (Zero=0.0d0)
C
      NumNZR = 1
      Do 20 I = 2, NTot
        I1 = 0
        Do 10 J = 1, NR
          If(X(J,I).ne.Zero) I1 = 1
   10     Continue
   20   NumNZR = NumNZR + I1
      Return
      End
*Deck Rd_2E1
      Subroutine Rd_2E1(IU,LR,NTot,LenBuf,RArr)
      Implicit None
      Integer IU,LR,NTot,LenBuf,IBuf(4,LenBuf),I,Ind,NDo,IJ,KL,IJKL
      Real*8 Buf(LenBuf),RArr(LR),Zero
      Parameter (Zero=0.0d0)
CF2PY Intent (Out) RArr
C
      Do 20 I = 1, LR
   20   RArr(I) = Zero
      Do 50 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Read(IU) IBuf, Buf
        Do 40 I = 1, NDo
          If(IBuf(1,I).ge.IBuf(2,I)) then
            IJ = (IBuf(1,I)*(IBuf(1,I)-1))/2 + IBuf(2,I)
          else
            IJ = (IBuf(2,I)*(IBuf(2,I)-1))/2 + IBuf(1,I)
            endIf
          If(IBuf(3,I).ge.IBuf(4,I)) then
            KL = (IBuf(3,I)*(IBuf(3,I)-1))/2 + IBuf(4,I)
          else
            KL = (IBuf(4,I)*(IBuf(4,I)-1))/2 + IBuf(3,I)
            endIf
          If(IJ.ge.KL) then
            IJKL = (IJ*(IJ-1))/2 + KL
          else
            IJKL = (KL*(KL-1))/2 + IJ
            endIf
   40     RArr(IJKL) = Buf(I)
   50   Continue
      Return
      End
*Deck Rd_2EN
      Subroutine Rd_2EN(IU,NR,LR,LRNR,NTot,LenBuf,RArr)
      Implicit None
      Integer IU,NR,LR,NTot,LenBuf,IBuf(4,LenBuf),I,J,Ind,NDo,IJ,KL,
     $  IJKL,LRNR
      Real*8 Buf(NR,LenBuf),RArr(LRNR),Zero
      Parameter (Zero=0.0d0)
CF2PY Intent (Out) RArr
C
      Do 10 I = 1, LRNR
   10   RArr(I) = Zero
      Do 50 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Read(IU) IBuf, Buf
        Do 40 I = 1, NDo
          If(IBuf(1,I).ge.IBuf(2,I)) then
            IJ = (IBuf(1,I)*(IBuf(1,I)-1))/2 + IBuf(2,I)
          else
            IJ = (IBuf(2,I)*(IBuf(2,I)-1))/2 + IBuf(1,I)
            endIf
          If(IBuf(3,I).ge.IBuf(4,I)) then
            KL = (IBuf(3,I)*(IBuf(3,I)-1))/2 + IBuf(4,I)
          else
            KL = (IBuf(4,I)*(IBuf(4,I)-1))/2 + IBuf(3,I)
            endIf
          If(IJ.ge.KL) then
            IJKL = (IJ*(IJ-1))/2 + KL
          else
            IJKL = (KL*(KL-1))/2 + IJ
            endIf
          Do 30 J = 1, NR
   30       RArr(IJKL+(J-1)*LR) = Buf(J,I)
   40     Continue
   50   Continue
      Return
      End
*Deck Rd_CBuf
      Subroutine Rd_CBuf(IU,NTot,LenBuf,Arr)
      Implicit None
      Integer IU,NTot,LenBuf,Ind,NDo,I
      Complex*16 Buf(LenBuf),Arr(NTot)
CF2PY Intent (Out) Arr
C
      Do 20 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Read(IU) Buf
        Do 10 I = 1, NDo
   10     Arr(Ind+I) = Buf(I)
   20   Continue
      Return
      End
*Deck Rd_Head
      Subroutine Rd_Head(IU,NLab,NAtoms,NBasis,IAn,IAtTyp,AtmChg,C,
     $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,NShlDB,
     $  NPrmDB,NBTot)
      Implicit None
      Integer IU,NLab,NAtoms,NBasis,IAn(NAtoms),IAtTyp(NAtoms),
     $  IBfAtm(NBasis),IBfTyp(NBasis),NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,
     $  NShlDB,NPrmDB,NBTot,I
      Real*8 AtmChg(NAtoms),C(3*NAtoms),AtmWgt(NAtoms)
CF2PY Intent(Out) IAn,IAtTyp,AtmChg,C,IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran
CF2PY Intent(Out) IDum9,NShlAO,NPrmAO,NShlDB,NPrmDB,NBTot
C
      NFC = 0
      NFV = 0
      ITran = 0
      IDum9 = 0
      NShlAO = 0
      NPrmAO = 0
      NShlDB = 0
      NPrmDB = 0
      NBTot = 0
      Call IClear(NAtoms,IAn)
      If(NLab.ge.3) Read(IU) IAn
      Call IClear(NAtoms,IAtTyp)
      If(NLab.ge.4) Read(IU) IAtTyp
      Call AClear(NAtoms,AtmChg)
      If(NLab.ge.5) Read(IU) AtmChg
      Call AClear(3*NAtoms,C)
      If(NLab.ge.6) Read(IU) (C(I),I=1,3*NAtoms)
      Call IClear(NBasis,IBfAtm)
      Call IClear(NBasis,IBfTyp)
      If(NLab.ge.7) Read(IU) IBfAtm,IBfTyp
        Do 10 I = 1, NAtoms
   10     AtmWgt(I) = 0.0d0
      If(NLab.ge.8) then
        Read(IU) AtmWgt
        If(NLab.ge.9) then
          Read(IU) NFC,NFV,ITran,IDum9
          If(NLab.ge.11) then
            Read(IU)
            Read(IU) NShlAO,NPrmAO,NShlDB,NPrmDB,NBTot
            Do 20 I = 12, NLab
              Read(IU)
   20         Continue
            endIf
          endIf
        endIf
      Return
      End
*Deck Rd_IBuf
      Subroutine Rd_IBuf(IU,NTot,LenBuf,Arr)
      Implicit None
      Integer IU,NTot,LenBuf,Buf(LenBuf),Arr(NTot),Ind,NDo,I
CF2PY Intent (Out) Arr
C
      Do 20 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Read(IU) Buf
        Do 10 I = 1, NDo
   10     Arr(Ind+I) = Buf(I)
   20   Continue
      Return
      End
*Deck Rd_Labl
      Subroutine Rd_Labl(IU,IVers,CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,
     $  ASym,NRI,EOF)
      Implicit None
      Integer IVers,LStr,IU,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,IASym,NRI
      Parameter (LStr=64)
C     The latest f2py is now "improved" and is now too stupid to handle
C     character string lengths given by parameters.
C     Character*(LStr) CBuf
      Character*64 CBuf
      Logical EOF,ASym
CF2PY Intent (Out) NI,NR,NRI,NTot,LenBuf,N1,N2,N3,N4,N5,ASym,NRI,EOF
CF2PY Intent (Out) CBuf
C
      CBuf = ' '
      NI = 0
      NR = 0
      NTot = 0
      LenBuf = 0
      N1 = 0
      N2 = 0
      N3 = 0
      N4 = 0
      N5 = 0
      IASym = 0
      If(IVers.eq.1) then
        Read(IU,End=900) CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5
      else
        Read(IU,End=900) CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,IASym
        endIf
      ASym = IASym.eq.-1
      EOF = CBuf.eq.'END'
      If(NR.ge.0) then
        NRI = 1
      else
        NRI = 2
        NR = -NR
        endIf
      If(.not.EOF) then
        If(N2.eq.0) N2 = 1
        If(N3.eq.0) N3 = 1
        If(N4.eq.0) N4 = 1
        If(N5.eq.0) N5 = 1
        endIf
      Return
  900 CBuf = 'END'
      EOF = .True.
      NI = 0
      NR = 0
      NRI = 1
      NTot = 0
      LenBuf = 0
      N1 = 0
      N2 = 0
      N3 = 0
      N4 = 0
      N5 = 0
      ASym = .False.
      Return
      End
*Deck Rd_RBuf
      Subroutine Rd_RBuf(IU,NTot,LenBuf,Arr)
      Implicit None
      Integer IU,NTot,LenBuf,Ind,NDo,I
      Real*8 Buf(LenBuf),Arr(NTot)
CF2PY Intent (Out) Arr
C
      Do 20 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Read(IU) Buf
        Do 10 I = 1, NDo
   10     Arr(Ind+I) = Buf(I)
   20   Continue
      Return
      End
*Deck Rd_RInd
      Subroutine Rd_RInd(IU,NR,LR,NTot,LenBuf,LNZ,RArr)
      Implicit None
      Integer IU,NR,LR,NTot,LenBuf,LNZ,I,J,Ind,NDo,IO,IBuf(LenBuf)
      Real*8 Buf(NR,LenBuf),RArr(NR,LR),Zero
      Parameter (Zero=0.0d0)
CF2PY Intent (Out) LNZ,RArr
C
      Do 20 I = 1, LR
        Do 10 J = 1, NR
   10     RArr(J,I) = Zero
   20   Continue
      LNZ = 1
      Do 50 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        If(NDo.eq.LenBuf.or..True.) then
          Read(IU) IBuf, Buf
        else
C         For debugging
          Read(IU) (IBuf(I),I=1,NDo), ((Buf(J,I),J=1,NR),I=1,NDo)
          endIf
        Do 40 I = 1, NDo
          IO = IBuf(I)
          LNZ = Max(LNZ,IO)
          Do 30 J = 1, NR
   30       RArr(J,IO) = Buf(J,I)
   40     Continue
   50   Continue
      Return
      End
*Deck Rd_Skip
      Subroutine Rd_Skip(IU,NTot,LenBuf)
      Implicit None
      Integer IU,NTot,LenBuf,I,NRec
C
      If(NTot.eq.0) Return
      NRec = (NTot+LenBuf-1)/LenBuf
      Do 10 I = 1, NRec
        Read(IU)
   10   Continue
      Return
      End
*Deck Wr_2E
      Subroutine Wr_2E(IU,NTot,NR,N,LR,LenBuf,RArr)
      Implicit None
      Logical NonZ
      Integer IU,NR,N,LR,LenBuf,IBuf(4,LenBuf),I,J,K,L,LimL,IJKL,
     $  IB,IR,NNZ,NTot
      Real*8 RArr(LR,NR),RBuf(NR,LenBuf),Zero
      Parameter (Zero=0.0d0)
C
      NNZ = 0
      IB = 0
      IJKL = 0
      Do 60 I = 1, N
        Do 50 J = 1, I
          Do 40 K = 1, I
            If(I.eq.K) then
              LimL = J
            else
              LimL = K
              endIf
            Do 30 L = 1, LimL
              IJKL = IJKL + 1
              NonZ = RArr(IJKL,1).ne.Zero
              Do 10 IR = 2, NR
   10           NonZ = NonZ.or.RArr(IJKL,IR).ne.Zero
              If(NonZ) then
                IB = IB + 1
                IBuf(1,IB) = I
                IBuf(2,IB) = J
                IBuf(3,IB) = K
                IBuf(4,IB) = L
                Do 20 IR = 1, NR
   20             RBuf(IR,IB) = RArr(IJKL,IR)
                If(IB.eq.LenBuf) then
                  Write(IU) IBuf, RBuf
                  NNZ = NNZ + LenBuf
                  IB = 0
                  endIf
                endIf
   30         Continue
   40       Continue
   50     Continue
   60   Continue
      If(IB.gt.0) then
        Do 80 I = (IB+1), LenBuf
          IBuf(1,I) = 0
          IBuf(2,I) = 0
          IBuf(3,I) = 0
          IBuf(4,I) = 0
          Do 70 IR = 1, NR
   70       RBuf(IR,I) = Zero
   80     Continue
        Write(IU) IBuf, RBuf
        NNZ = NNZ + IB
        endIf
      If(NNZ.ne.NTot) Stop 'NNZ not NTot in Wr_2E'
      Return
      End
*Deck Wr_CBuf
      Subroutine Wr_CBuf(IU,NTot,LenBuf,Arr)
      Implicit None
      Integer IU,NTot,LenBuf,Ind,NDo,I
      Complex*16 Buf(LenBuf),Arr(NTot),Zero
C     The latest f2py is now "improved" and is now too stupid to handle
C     complex variables in parameter statements.
C     Parameter (Zero=(0.0d0,0.0d0))
      Real*8 ZeroR
      Parameter (ZeroR=0.0d0)
C
      Zero = DCmplx(ZeroR,ZeroR)
      Do 30 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Do 10 I = 1, NDo
   10     Buf(I) = Arr(Ind+I)
        Do 20 I = (NDo+1), LenBuf
   20     Buf(I) = Zero
        Write(IU) Buf
   30   Continue
      Return
      End
*Deck Wr_Head
      Subroutine Wr_Head(IU,NAtoms,NAt3,NBasis,IAn,IAtTyp,AtmChg,C,
     $  IBfAtm,IBfTyp,AtmWgt,NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,NShlDB,
     $  NPrmDB,NBTot)
      Implicit None
      Integer LRec11,IZero
      Parameter (LRec11=16,IZero=0)
      Integer IU,NAtoms,NAt3,NBasis,IAn(NAtoms),IAtTyp(NAtoms),
     $  IBfAtm(NBasis),IBfTyp(NBasis),NFC,NFV,ITran,IDum9,NShlAO,NPrmAO,
     $  NShlDB,NPrmDB,NBTot,I,Rec11(LRec11),LRec(2)
      Real*8 AtmChg(NAtoms),C(NAt3),AtmWgt(NAtoms)
C
      If(Mod(NAtoms,2).eq.1) then
        Write(IU) IAn,IZero
        Write(IU) IAtTyp,IZero
      else
        Write(IU) IAn
        Write(IU) IAtTyp
        endIf
      Write(IU) AtmChg
      Write(IU) (C(I),I=1,3*NAtoms)
      Write(IU) IBfAtm,IBfTyp
      Write(IU) AtmWgt
      Write(IU) NFC,NFV,ITran,IDum9
      LRec(1) = LRec11
      LRec(2) = 0
      Write(IU) LRec
      Do 10 I = 1, LRec11
   10   Rec11(I) = 0
      Rec11(1) = NShlAO
      Rec11(2) = NPrmAO
      Rec11(3) = NShlDB
      Rec11(4) = NPrmDB
      Rec11(5) = NBTot
      Write(IU) Rec11
      Return
      End
*Deck Wr_IBuf
      Subroutine Wr_IBuf(IU,NTot,LenBuf,Arr)
      Implicit None
      Integer IU,NTot,LenBuf,Buf(LenBuf),Arr(NTot),Ind,NDo,I
C
      Do 30 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Do 10 I = 1, NDo
   10     Buf(I) = Arr(Ind+I)
        Do 20 I = (NDo+1), LenBuf
   20     Buf(I) = 0
        Write(IU) Buf
   30   Continue
      Return
      End
*Deck Wr_Labl
      Subroutine Wr_Labl(IU,CBuf,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,ASym)
      Implicit None
      Logical ASym
      Integer LStr,IU,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,IASym
      Parameter (LStr=64)
C     The latest f2py is now "improved" and is now too stupid to handle
C     character string lengths given by parameters.
C     Character CBuf*(*), CBufL*(LStr)
      Character CBuf*(*), CBufL*64
C
      If(ASym) then
        IASym = -1
      else
        IASym = 0
        endIf
      CBufL = CBuf
      Write(IU) CBufL,NI,NR,NTot,LenBuf,N1,N2,N3,N4,N5,IASym
      Return
      End
*Deck Wr_RBuf
      Subroutine Wr_RBuf(IU,NTot,LenBuf,Arr)
      Implicit None
      Integer IU,NTot,LenBuf,Ind,NDo,I
      Real*8 Buf(LenBuf),Arr(NTot),Zero
      Parameter (Zero=0.0d0)
C
      Do 30 Ind = 0, (NTot-1), LenBuf
        NDo = Min(LenBuf,NTot-Ind)
        Do 10 I = 1, NDo
   10     Buf(I) = Arr(Ind+I)
        Do 20 I = (NDo+1), LenBuf
   20     Buf(I) = Zero
        Write(IU) Buf
   30   Continue
      Return
      End
*Deck Wr_RInd
      Subroutine Wr_RInd(IU,NR,LR,NTot,LenBuf,RArr)
      Implicit None
      Logical NonZer
      Integer IU,NR,LR,NTot,LenBuf,I,J,IBuf(LenBuf),NNZ,IB
      Real*8 Buf(NR,LenBuf),RArr(NR,LR),Zero
      Parameter (Zero=0.0d0)
C
      NNZ = 0
      IB = 0
      Do 40 I = 1, LR
        NonZer = I.eq.1
        Do 30 J = 1, NR
          Buf(J,IB+1) = RArr(J,I)
   30     NonZer = NonZer.or.Buf(J,IB+1).ne.Zero
        If(NonZer) then
          IB = IB + 1
          IBuf(IB) = I
          If(IB.eq.LenBuf) then
            Write(IU) IBuf, Buf
            IB = 0
            NNZ = NNZ + LenBuf
            endIf
          endIf
   40   Continue
      If(IB.gt.0) then
        NNZ = NNZ + IB
        Do 60 I = (IB+1), LenBuf
          IBuf(I) = 0
          Do 50 J = 1, NR
   50       Buf(J,I) = Zero
   60     Continue
        Write(IU) IBuf, Buf
        endIf
      If(NNZ.ne.NTot) Stop 'NZ error in Wr_RInd'
      Return
      End
*Deck ExpAO1
      Subroutine ExpAO1(N,LR,RI,RO)
      Implicit None
      Integer N,LR,I,J,K,L,LimL,IJKL
      Real*8 RI(LR), RO(N,N,N,N), R
CF2PY Intent (Out) RO
C
      IJKL = 0
      Do 40 I = 1, N
        Do 30 J = 1, I
          Do 20 K = 1, I
            If(I.eq.K) then
              LimL = J
            else
              LimL = K
              endIf
            Do 10 L = 1, LimL
              R = RI(IJKL+L)
              RO(I,J,K,L) = R
              RO(I,J,L,K) = R
              RO(J,I,K,L) = R
              RO(J,I,L,K) = R
              RO(K,L,I,J) = R
              RO(L,K,I,J) = R
              RO(K,L,J,I) = R
   10         RO(L,K,J,I) = R
   20       IJKL = IJKL + LimL
   30     Continue
   40   Continue
      Return
      End
*Deck ExpAON
      Subroutine ExpAON(NE,N,LR,RI,RO)
      Implicit None
      Integer N,LR,I,J,K,L,LimL,IJKL,IE,NE
      Real*8 RI(NE,LR), RO(NE,N,N,N,N), R
CF2PY Intent (Out) RO
C
      IJKL = 0
      Do 40 I = 1, N
        Do 30 J = 1, I
          Do 20 K = 1, I
            If(I.eq.K) then
              LimL = J
            else
              LimL = K
              endIf
            Do 10 L = 1, LimL
              Do 5 IE = 1, NE
                R = RI(IE,IJKL+L)
                RO(IE,I,J,K,L) = R
                RO(IE,I,J,L,K) = R
                RO(IE,J,I,K,L) = R
                RO(IE,J,I,L,K) = R
                RO(IE,K,L,I,J) = R
                RO(IE,L,K,I,J) = R
                RO(IE,K,L,J,I) = R
    5           RO(IE,L,K,J,I) = R
   10         Continue
   20       IJKL = IJKL + LimL
   30     Continue
   40   Continue
      Return
      End
*Deck AClear
      Subroutine AClear(N,A)
      Implicit None
C
C     Routine to clear N elements in array A.
C
      Integer N, I
      Real*8 Zero, A(N)
      Parameter (Zero=0.0D0)
C
      Do 10 I = 1, N
   10   A(I) = Zero
      Return
      End
*Deck IClear
      Subroutine IClear(N,IA)
      Implicit None
C
C     Routine to clear N elements in array IA.
C
      Integer N,IA(N),I
C
      Do 10 I = 1, N
   10   IA(I) = 0
      Return
      End
