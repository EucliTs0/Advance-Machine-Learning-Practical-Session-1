classdef simpleModelSVDD < simpleModel
    % SIMPLEMODELCSVM This class inherits from simpleModel. It contains
    % particular methods to solve Support Vector Data Description
    % This class support SVDD solver, warm-start, 2D solution plot, and
    % cross-validation error-rate evaluation (see simpleModel.crossValError).
    % See also : simpleModel, simpleData
    
    % Create by Gaëlle Loosli
    % 29/11/2013
    % Last Modified 29/11/2013 by G.L.
    
    
    
    methods
        % contructor
        function SM = simpleModelSVDD(SD,C,Il,Iu,Iw)
            % simpleModelSVDD : constructor.
            % Input :   SD : a simpleData object
            %           C : Slack value (can be of length 1 for all points
            %           equally penalized, or one value by class, or one
            %           value per training point)
            %           Il : indices list of non support vectors for warm-start
            %           Iu : indices list of upper bounded support vectors for
            %           warm-start
            %           Iw : indices list of non-bounded support vectors
            % Number of input arguments must be at least 2 (simpleData
            % object and simpleProblem object) or exactly 5 (for warm-start,
            % one can provide the initial repartition of support vectors)
            % Output :   SM : a simpleModelSVDD object
            
            assert(nargin==2 || nargin==5,'simpleModelSVDD:wrongInput','Requires 2 or 5 input arguments');
            if nargin<2
                error('not enough input');
            else
                SD.getKernelValues(1:SD.n);  % pre-compute kernel
                %SD.kernelCache(1:SD.n,1:SD.n) = SD.kernelCache(1:SD.n,1:SD.n) + 0.0001*eye(SD.n);
                SM.kktgap=10^-5;
                SM.SD = SD;
                SM.C = C;
                %SM.Slacks = zeros(SD.n,1);
                if nargin==5
                    SM.Il = Il;
                    SM.Iu = Iu;
                    SM.Iw = Iw;
                    SM.oldalpha = zeros(length(Iw),1);
                else
                    SM.Iw = [1,2];
                    SM.Iu = [];
                    SM.Il = 1:SD.n;
                    SM.Il = setdiff(SM.Il,SM.Iw);
                    SM.oldalpha = zeros(length(SM.Iw),1);
                end
                SM.labs = sort(unique(SD.trainlab));
                if length(C)==length(SD.trainlab)
                    SM.Slacks = C;
                else
                    SM.Slacks = C(1)*ones(length(SD.trainlab),1);
                end
                SM.activated = 0;
                SM.converged = 0;
            end
        end
        
        
        
        
        
        function solveLinearSystem(obj)
            
            
            be = -1;
            indd = obj.SD.getKernelValues(obj.Iw,obj.Iu);
            ce = diag(obj.SD.kernelCache(obj.Iw,indd));
            if ~isempty(obj.Iu)
                ce = ce - 2*((obj.Slacks(obj.Iu))'*obj.SD.kernelCache(obj.Iu,indd))';
                be = be + obj.Slacks(obj.Iu)'*obj.SD.trainlab(obj.Iu);
            end
            if obj.optimType==1
                G = chol(obj.SD.kernelCache(obj.Iw,indd)+obj.kktgap*eye(length(obj.Iw)));
                M = obj.SD.trainlab(obj.Iw)'*(G\(G'\obj.SD.trainlab(obj.Iw)));
                obj.b = M\(obj.SD.trainlab(obj.Iw)'*(G\(G'\ce))+2* be);
                obj.alpha = 0.5*( G\(G'\(ce-obj.SD.trainlab(obj.Iw)*obj.b)));
%                 
%                 while(obj.b<=0)
%                     be = be+0.02;
%                     obj.b = M\(obj.SD.trainlab(obj.Iw)'*(G\(G'\ce))+2*be);
%                     obj.alpha = 0.5*( G\(G'\(ce-obj.SD.trainlab(obj.Iw)*obj.b)));
%                     %disp( ['--- ', num2str(obj.b), ', ', num2str(be),' , ', num2str(sum(obj.alpha)+obj.C*length(obj.Iu))]);
%                 end
                
                % disp( ['. ', num2str(obj.b), ', ', num2str(be),' , ' num2str(sum(obj.alpha)+obj.C*length(obj.Iu))]);
            else
                error('not implemented yet');
            end
        end
        
        
        function reinit(obj)
            obj.Iw = find(obj.SD.trainlab==1,2);
            obj.Iu = [];
            obj.Il = 1:obj.SD.n;
            obj.Il = setdiff(obj.Il,obj.Iw);
            obj.oldalpha = zeros(2,1);
            
            if length(obj.Slacks)~=length(obj.SD.trainlab)
                if length(obj.C)==length(obj.SD.trainlab)
                    obj.Slacks = obj.C;
                elseif length(obj.C)==length(obj.labs)
                    for i=1:length(obj.labs)
                        obj.Slacks(obj.SD.trainlab==obj.labs(i)) = obj.C(i);
                    end
                else
                    obj.Slacks = obj.C(1)*ones(size(obj.SD.trainlab));
                    
                end
            end
            obj.activated = 0;
            obj.converged = 0;
        end
        
        function print2D(obj)
            assert(size(obj.SD.trainvec,1)==2,'simpleModelSVDD:print2D','must be 2D data');
            
            global SMscale; if SMscale==[], SMscale=2; end
            global SMstep; if SMscale==[], SMscale=0.1; end
            [xtesta1,xtesta2]=meshgrid([-SMscale:SMstep:SMscale],[-SMscale:SMstep:SMscale]);
            [na,nb]=size(xtesta1);
            xtest1=reshape(xtesta1,1,na*nb);
            xtest2=reshape(xtesta2,1,na*nb);
            xtest=[xtest1;xtest2];
            obj.SV = obj.SD.trainvec(:,[obj.Iw,obj.Iu]);
            [ypred perf] = obj.test(xtest,ones(size(xtest,2),1));
            ypredmat=reshape(ypred,na,nb);
            
            hold off;
            plot(obj.SD.trainvec(1,:),obj.SD.trainvec(2,:),'r+');
            hold on;
            plot(obj.SV(1,:),obj.SV(2,:),'om','MarkerSize',10,'LineWidth',2);
            [c,h] = contour(xtesta1,xtesta2,ypredmat,[ 0 0],'g');
            set(h,'LineWidth',2);
            %[c,h] = contour(xtesta1,xtesta2,ypredmat,[ -obj.b obj.b],'y');
            %set(h,'LineWidth',2);
            drawnow;
            
        end
        
        
        function activateBoxConstraints(obj)
            if (~isempty(find(obj.alpha<0, 1)) || (~isempty(find(obj.alpha>obj.Slacks(obj.Iw), 1)) && length(obj.Iw)+length(obj.Iu)>1/obj.C))  && length(obj.Iw)>2
            
            %if (~isempty(find(obj.alpha<0, 1)) || (~isempty(find(obj.alpha>obj.Slacks(obj.Iw), 1))) ) && length(obj.Iw)>2
                dir = obj.alpha - obj.oldalpha;
                indad = find(obj.alpha < 0);
                indsup = find(obj.alpha > obj.Slacks(obj.Iw));
                [tI indmin] = min(-obj.oldalpha(indad)./dir(indad));
                [tS indS] = min((obj.Slacks(indsup)-obj.oldalpha(indsup))./dir(indsup));
                if isempty(tI) , tI = tS + 1; end;
                if isempty(tS) , tS = tI + 1; end;
                t = min(tI,tS);
                obj.oldalpha = obj.oldalpha + t*dir;
                if t==tI
                    obj.Il = sort([obj.Il , obj.Iw(indad(indmin))]);
                    obj.Iw(indad(indmin)) = [];
                    obj.oldalpha(indad(indmin)) = [];
                    obj.alpha(indad(indmin)) = [];
                else
                    obj.Iu = sort([obj.Iu , obj.Iw(indsup(indS))]);
                    obj.Iw(indsup(indS))= [];
                    obj.oldalpha(indsup(indS))= [];
                    obj.alpha(indsup(indS)) = [];
                end
                obj.activated = 1;
            else
                obj.activated = 0;
            end
        end
        
        function [y] = evaluateSolution(obj,candidates)
            indTR = obj.SD.getKernelValues(candidates,candidates);
            TR = diag(obj.SD.kernelCache(candidates,indTR));
            indd = obj.SD.getKernelValues([obj.Iw,obj.Iu],candidates);
            [val,inds]= sort(indd);
            temp = [obj.alpha;obj.Slacks(obj.Iu)];
            a = obj.SD.kernelCache(candidates,val)*temp(inds);
            y = obj.b - TR + 2*a;
        end
        
        function train(obj)
            iter = 0;
            while obj.converged==0 && iter<500*length(obj.SD.n)
                iter = iter+1;
                obj.solveLinearSystem();
                %obj.print2D();
                %keyboard
                obj.activateBoxConstraints();
                if (obj.activated==0)
                    obj.relaxBoxConstraints();
                end
            end
            if obj.converged==0
                disp('No convergence');
                obj.solveLinearSystem();
                obj.converged = 1;
            end
            obj.finish();
            %obj.print2D();
            
            
        end
        
        
        
        function relaxBoxConstraints(obj)
            
            %obj.iloop = 1;
            xt = zeros(obj.SD.n,1);
            xt(obj.Iw) = obj.alpha;
            xt(obj.Iu) = obj.Slacks(obj.Iu);
            S = 300;
            if (length(obj.Il)==500)
                S=450;
            end
            Found=0;
            nbLoop =ceil(length(obj.Il)/S);
            %if obj.iloop>=nbLoop, obj.iloop=1; end
            obj.iloop = mod(obj.iloop,nbLoop)+1;
            z = 0;
            if not(isempty(obj.Il))
                while (z<nbLoop && Found==0)
                    z = z+1;
                    nul = obj.evaluateSolution(obj.Il((obj.iloop-1)*S+1:min(obj.iloop*S,end)));
                    [mm mpos]=min(nul);
                    if (mm<-obj.kktgap)
                        Found=1;
                        mpos = mpos+(obj.iloop-1)*S;
                    else
                        mm = inf; mpos = [];
                    end
                    %obj.iloop= obj.iloop+1;
                    obj.iloop = mod(obj.iloop,nbLoop)+1;
                    %keyboard
                end
            else
                mm = inf; mpos = [];
            end
            if not(isempty(obj.Iu))
                nuu = obj.evaluateSolution(obj.Iu);
                [mmS mposS]=min(-nuu);
            else
                mmS = inf; mposS = [];
            end
            
            if (~isempty(mmS) && mmS < -obj.kktgap) || (~isempty(mm) && mm < -obj.kktgap)
                if isempty(mmS) || mm < mmS
                    obj.Iw = sort([obj.Iw , obj.Il(mpos)]);
                    obj.oldalpha = xt(obj.Iw);
                    obj.Il(mpos) = [];
                else
                    obj.Iw = sort([obj.Iw , obj.Iu(mposS)]);
                    obj.oldalpha = xt(obj.Iw);
                    obj.Iu(mposS)=[];
                end
            else
                obj.converged = 1;
            end
        end
        
        
        function [ypred perf] = test(obj,testdata,testlabel)
            %indd = obj.SD.getKernelValues(obj.Iw(1),obj.Iw(1));
            %R = obj.SD.kernelCache(obj.Iw(1),indd) - 2*obj.SD.kernelCache(obj.Iw(1),[obj.Iw,obj.Iu])*obj.alpha;
            %[R, obj.b]
            if strcmp(obj.SD.kernelType,'precomputed') % pre-computed case
                assert(size(testdata,1)==obj.SD.n,'simpleModelSVDD:wrongInput','precomputed-kernel used, need to provide a matrix nntrain*nbtest to compute a test error');
                ypred = obj.b - diag(testdata) + 2*(obj.alpha'*testdata([obj.Iw,obj.Iu],:))';
                if nargin==3
                    perf = 100*length(find(sign(ypred)==testlabel))/length(testlabel);
                else
                    perf = -1;
                end
            else
                K = kernelTest(obj.SD.kernelType,obj.SD.kernelParam,obj.SD.trainvec(:,[obj.Iw,obj.Iu]), ...
                    abs(obj.SD.trainlab([obj.Iw,obj.Iu])), testdata, ones(size(testdata,2),1));
                K2 =  kernelTest(obj.SD.kernelType,obj.SD.kernelParam,testdata, ...
                    ones(size(testdata,2),1), testdata, ones(size(testdata,2),1));
                ypred = obj.b - diag(K2) + 2*(K*obj.alpha);
                if nargin==3
                    perf = 100*length(find(sign(ypred)==testlabel))/length(testlabel);
                else
                    perf = -1;
                end
            end
        end
        
    end
    
end
