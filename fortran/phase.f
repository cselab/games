c     Nash multi-agent bargaining game. september 25th 2017
c     Regime transitions in beta. Monitoring agents' choices.
c     Moore neighborhood (4 neighbors)
c     gfortran -O3 transforjasss.f -ffixed-line-length-132 -finit-local-zero -fbounds-check -o AB
c     AB | DynamicLattice -nx 210 -ny 140 -z 1 3 -cmap mycol.map
      integer*8 period  ! iteration time
      parameter(L=33)   ! lattice linear dimensions
      integer seed,ix,iy,k,ij,ibeta ! random generator seed, coordinates, J index, beta loop index
      integer i,j(4),nr,nb,ng  ! J index, alter coordinates, to compute HLM fractions
      integer, DIMENSION(0:L+1,0:L+1) :: choice  ! agent's choice
      REAL*8, DIMENSION(0:L+1,0:L+1,0:4) :: bra,ech  ! J,Boltzmann exponential
      intrinsic mod,int
      real mat(3,3),lambda,gam,beta  ! payoff matrix, !!! 1-gamma, !!! lambda, beta
      real seb,J01,J02,J03  ! Boltzmann denominator * rand, max initial J1,J2,J3
      real L2 ! number of agents
      beta=3 ! Boltzmann coefficient
      L2=float(L*L)  !agents' number
      lambda=0.9 ! decrease factor of the moving average
      J01=4.0
      J02=4.0
      J03=4.0
      gam=0.2  ! inequity parameter
      period=10000000 !nb iterations
      OPEN (unit=13,file='fractions-data',status='unknown')
      mat(1,1)=0                ! payoff matrix
      mat(1,2)=0                !
      mat(1,3)=0.5+gam          !
      mat(2,1)=0                !
      mat(2,2)=0.5              !
      mat(2,3)=0.5              !
      mat(3,1)=0.5-gam          !
      mat(3,2)=0.5-gam          !
      mat(3,3)=0.5-gam          !
      seed=8531
      do ix=0,L-1  ! J coefficients initial random sampling
         do iy=0,L-1
            bra(ix,iy,1)=J01*ran2(seed) !H preference coefficient
            bra(ix,iy,2)=J02*ran2(seed) !M
            bra(ix,iy,3)=J03*ran2(seed) !L
            choice(ix,iy)=1    ! to avoid problems with Index '0'
         enddo
      enddo
cccccccccccccccccccccccccc
      do ibeta=1,60  ! loop on beta decrease
         beta=beta-0.05
         print *, beta
cccccccccccfin init ccccccccccccccccccccccc
         do itime=1,period      !main loop on time
            ix=int(L*ran2(seed)) !! random sampling of
            iy=int(L*ran2(seed)) ! ego coordinates! and now sampling for alter neighbor
            jx=ix               ! neighbor coordinate to be changed if  itir= 1 ou 2
            jy=iy               ! to be changed if itir= 3 or 4
            itir=int(1.0+3.9999*ran2(seed)) ! sampling for alter neighbor
            if (itir.eq.1) then
               jx=mod(ix+1,L) ! right neighbor
            endif
            if (itir.eq.2) then
              jx=mod(ix+L-1,L) !  left neighbor
            endif
            if (itir.eq.3) then
              jy=mod(iy+1,L) ! up neighbor
            endif
            if (itir.eq.4) then
              jy=mod(iy+L-1,L) ! down neighbor
            endif
            j(1)=ix             !ego
            j(2)=iy             !ego
            j(3)=jx             !alter
            j(4)=jy             !alter
            do k=1,3,2          !! Boltzmann loop on ego and alter
              seb=0            ! to figure out their choices
              do ij=1,3        ! J index
                ech(j(k),j(k+1),ij)=exp(beta*bra(j(k),j(k+1),ij)) ! Boltzmann exponential
                seb=seb+ech(j(k),j(k+1),ij)
              enddo
              rs=ran2(seed)*seb
              if ( rs .lt. ech(j(k),j(k+1),1) ) then
                choice(j(k),j(k+1))=1
              else if (rs.lt.(ech(j(k),j(k+1),1)+ech(j(k),j(k+1),2)))
     &        then
                choice(j(k),j(k+1))=2
              else
                choice(j(k),j(k+1))=3
              endif

         enddo ! end of Boltzmann loop ! beginning of J moving average updating
         do i=1,3
            bra(ix,iy,i)=lambda*bra(ix,iy,i)
            bra(jx,jy,i)=lambda*bra(jx,jy,i)
         enddo
         bra(ix,iy,choice(ix,iy))=bra(ix,iy,choice(ix,iy))
     &                              +mat(choice(ix,iy),choice(jx,jy))
         bra(jx,jy,choice(jx,jy))=bra(jx,jy,choice(jx,jy))
     &                              +mat(choice(jx,jy),choice(ix,iy))
      enddo                     ! end on the updating loop on time

      nr=0  ! summing agents choices
      nb=0
      ng=0
      do ix=0,L-1 ! Computing HLM choice fractions for display
         do iy=0,L-1
            if (choice(ix,iy).EQ.1) then ! H choice
               nr=nr+1
            endif
            if (choice(ix,iy).EQ.2) then ! M choice
               nb=nb+1
            endif
            if (choice(ix,iy).EQ.3) then ! L choice
               ng=ng+1
            endif
         enddo
      enddo                     ! end of the lattice loop
      write (13,*) beta, nr/L2,nb/L2,ng/L2 ! beta, fractions of agents' choices
      enddo  ! end of the loop on beta
      close(13)
      END
c=====random number generator from Numerical Recipes==========
      FUNCTION RAN2(IDUM)
      PARAMETER (M=714025,IA=1366,IC=150889,RM=1.4005112E-6)
      DIMENSION IR(97)
      DATA IFF /0/
      SAVE
      IF(IDUM.LT.0.OR.IFF.EQ.0)THEN
        IFF=1
        IDUM=MOD(IC-IDUM,M)
        DO 11 J=1,97
          IDUM=MOD(IA*IDUM+IC,M)
          IR(J)=IDUM
11      CONTINUE
        IDUM=MOD(IA*IDUM+IC,M)
        IY=IDUM
      ENDIF
      J=1+(97*IY)/M
      IF(J.GT.97.OR.J.LT.1) THEN
        PAUSE
      ENDIF
      IY=IR(J)
      RAN2=IY*RM
      IDUM=MOD(IA*IDUM+IC,M)
      IR(J)=IDUM
      RETURN
      END
