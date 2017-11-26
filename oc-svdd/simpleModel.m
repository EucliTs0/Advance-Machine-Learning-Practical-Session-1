classdef simpleModel < handle
    % SIMPLEMODEL This class contains common part of all SVM solvers. It
    % is abstract (can't be instanciated).
    % See also : simpleModelCSVM , simpleModelNuSVM
    
    % Create by Ga�lle Loosli
    % 16/02/2011
    % Last Modified 16/02/2011 by G.L.
    
    properties
        labs; % list of data labels
        kktgap = 10^-6; % KKT Gap for stopping criteria
        optimType = 1; % 1 : chol, 2 : QR
        alpha; % SV Lagrange multipliers
        b; % Solution bias
        Il; % List of indices of lower bounded points
        Iu; % List of indices of upper bounded points
        Iw; % List of indices of non bounded points
        SV; % Final list of Support Vectors (vectors), empty if the provided kernel matrix was pre-computed.
        Slacks; % List of slack variables
        SD; % data
        oldalpha; % previous admissible solution
        converged; % true if the algorithm has converges
        activated; % true if a contraint has be activated
        C;
        iloop = 1;
        cout = [];
    end
    
    
    methods (Abstract)
        
        solveLinearSystem(obj)
        reinit(obj)
        
    end
    
    methods
        function set.labs(obj,val)
            obj.labs = val;
        end
        
        function set.C(obj,val)
            obj.C = val;
        end
        
        function set.kktgap(obj,val)
            assert(val>=0,'simpleProble:wrongInput','kkt gap must be positive');
            obj.kktgap = val;
        end
        
        function set.optimType(obj,type)
            assert(val==1 || val==2 ,'simpleProble:wrongInput','optimtype must be 1 (chol) or 2 (QR)');
            obj.optimType = type;
        end
        
        function val = get.labs(obj)
            val = obj.labs;
        end
        
        
        function set.iloop(obj,val)
            obj.iloop = val;
        end
        
        function i = get.iloop(obj)
            i = obj.iloop;
        end
        
        
        function val = get.C(obj)
            val = obj.C;
        end
        
        function val = get.kktgap(obj)
            val = obj.kktgap;
        end
        
        function type = get.optimType(obj)
            type = obj.optimType;
        end
        
        
        function set.alpha(obj,val)
            obj.alpha = val;
        end
        
        function val = get.alpha(obj)
            val = obj.alpha;
        end
        
        function set.b(obj,val)
            obj.b = val;
        end
        
        function val = get.b(obj)
            val = obj.b;
        end
        
        function set.Il(obj,val)
            obj.Il = val;
        end
        
        function val = get.Il(obj)
            val = obj.Il;
        end
        
        function set.Iu(obj,val)
            obj.Iu = val;
        end
        
        function val = get.Iu(obj)
            val = obj.Iu;
        end
        
        function set.Iw(obj,val)
            obj.Iw = val;
        end
        
        function val = get.Iw(obj)
            val = obj.Iw;
        end
        
        function set.SV(obj,val)
            obj.SV = val;
        end
        
        function val = get.SV(obj)
            val = obj.SV;
        end
        
        function set.Slacks(obj,val)
            obj.Slacks = val;
        end
        
        function val = get.Slacks(obj)
            val = obj.Slacks;
        end
        
        function set.oldalpha(obj,val)
            obj.oldalpha = val;
        end
        
        function val = get.oldalpha(obj)
            val = obj.oldalpha;
        end
        
        function set.converged(obj,val)
            obj.converged = val;
        end
        
        function val = get.converged(obj)
            val = obj.converged;
        end
        
        function set.activated(obj,val)
            obj.activated = val;
        end
        
        function val = get.activated(obj)
            val = obj.activated;
        end
        
        
        function set.cout(obj,val)
            obj.cout = val;
        end
        
        function val = get.cout(obj)
            val = obj.cout;
        end
        
        function y = evaluateSolution(obj,candidates)
            indd = obj.SD.getKernelValues([obj.Iw,obj.Iu],candidates);
            [val,inds]= sort(indd);
            temp = [obj.alpha;obj.Slacks(obj.Iu)];
            y = obj.SD.kernelCache(candidates,val)*temp(inds)+obj.SD.trainlab(candidates)*obj.b;
        end
        
        function clear(obj)
            obj.alpha = [];
            obj.SV = [];
            obj.b = [];
        end
        
        
        
        function err = crossValError(obj,nbFold)
            assert(nbFold<=obj.SD.n && nbFold>0,'simpleModel:wrongIput','nbFold must range between 1 and number of training points');
            Sl = obj.Slacks;
            SDSaved = obj.SD;
                s = ceil(obj.SD.n/nbFold);
                p = zeros(1,nbFold);
                rn = randperm(SDSaved.n);
            if ~isempty(obj.SD.trainvec)
                for i=1:nbFold
                    il = rn((i-1)*s+1:min(i*s,SDSaved.n));
                    it = setdiff(1:SDSaved.n,il);
                    if strcmp(SDSaved.kernelType,'precomputed')
                        SDt = simpleData(SDSaved.trainvec(it,it),SDSaved.trainlab(it));
                    else
                        SDt = simpleData(SDSaved.trainvec(:,it),SDSaved.trainlab(it));
                    end
                    SDt.kernelType = SDSaved.kernelType;
                    SDt.kernelParam = SDSaved.kernelParam;
                    obj.SD = SDt;
                    obj.Slacks = Sl(it);
                    obj.clear();
                    obj.reinit();
                    obj.train();
                    if strcmp(SDSaved.kernelType,'precomputed')
                        [~ , p(i)] = obj.test(SDSaved.trainvec(it,il),SDSaved.trainlab(il));
                    else
                        [~ , p(i)] = obj.test(SDSaved.trainvec(:,il),SDSaved.trainlab(il));
                    end
                   clear SDt;
                end
            else
                SDSaved.kernelCache = SDSaved.kernelCache.*(SDSaved.trainlab*SDSaved.trainlab');
                
                for i=1:nbFold
                    il = rn((i-1)*s+1:min(i*s,SDSaved.n));
                    it = setdiff(1:SDSaved.n,il);
                    SDt = simpleData(SDSaved.kernelCache(it,it),SDSaved.trainlab(it));
                    obj.SD = SDt;
                    obj.Slacks = Sl(it);
                    obj.clear();
                    obj.reinit();
                    obj.train();
                    [~ , p(i)] = obj.test(SDSaved.kernelCache(it,il),SDSaved.trainlab(il));
                    clear SDt;
                end
                SDSaved.kernelCache = SDSaved.kernelCache.*(SDSaved.trainlab*SDSaved.trainlab');
            end
            err = 100 - mean(p);
            obj.SD = SDSaved();
            obj.Slacks = Sl;
        end
        
        
        function train(obj)
            iter = 0;
            while obj.converged==0 && iter<3000*obj.SD.n
                iter = iter+1;
                obj.solveLinearSystem();
                %obj.print2D(); 
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
        
        function activateBoxConstraints(obj)
            if (~isempty(find(obj.alpha<0, 1)) || ~isempty(find(obj.alpha>obj.Slacks(obj.Iw), 1))) && length(obj.alpha)>2
                dir = obj.alpha - obj.oldalpha;
                indad = find(obj.alpha < 0);
                indsup = find(obj.alpha > obj.Slacks(obj.Iw));
                [tI indmin] = min(-obj.oldalpha(indad)./dir(indad));
                [tS indS] = min((obj.Slacks(obj.Iw(indsup))-obj.oldalpha(indsup))./dir(indsup));
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
        
        
        function relaxBoxConstraints(obj)
            xt = zeros(obj.SD.n,1);
            xt(obj.Iw) = obj.alpha;
            xt(obj.Iu) = obj.Slacks(obj.Iu);
            S = 500;
            if (length(obj.Il)==500)
                S=450;
            end
            Found=0;
            nbLoop = ceil(length(obj.Il)/S);
            z = 0;
            if not(isempty(obj.Il))
                while (z<nbLoop && Found==0)
                    z = z+1;
                    nul = obj.evaluateSolution(obj.Il((obj.iloop-1)*S+1:min(obj.iloop*S,end)))-1;
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
                nuu = obj.evaluateSolution(obj.Iu)-1;
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
        
        function finish(obj)
            if ~isempty(obj.SD.trainvec)
                obj.SV = obj.SD.trainvec(:,[obj.Iw,obj.Iu]);
                
            else
                %disp('no training data can be stored for later use :
                %pre-computed matrix');
            end
            obj.alpha = [obj.alpha;obj.Slacks(obj.Iu)].*obj.SD.trainlab([obj.Iw,obj.Iu]);
            
        end
        
        function [ypred perf] = test(obj,testdata,testlabel)
            if strcmp(obj.SD.kernelType,'precomputed') % pre-computed case
                assert(size(testdata,1)==obj.SD.n,'simpleModel:wrongInput','precomputed-kernel used, need to provide a matrix nntrain*nbtest to compute a test error');
                ypred = (obj.alpha'*testdata([obj.Iw,obj.Iu],:)+obj.b)';
                if nargin==3
                    perf = 100*length(find(sign(ypred)==testlabel))/length(testlabel);
                else
                    perf = -1;
                end
            else
                %ypred = (obj.alpha'*kernelEval(obj,testdata)+obj.b)';
                ypred = (obj.alpha'*kernelTest(obj.SD.kernelType,obj.SD.kernelParam,obj.SD.trainvec(:,[obj.Iw,obj.Iu]), ...
                    abs(obj.SD.trainlab([obj.Iw,obj.Iu])), testdata, ones(size(testdata,2),1))'+obj.b)';
                if nargin==3
                    perf = 100*length(find(sign(ypred)==testlabel))/length(testlabel);
                else
                    perf = -1;
                end
            end
        end
        
        
        function SMt = train1vs1(obj,SMt)
            assert(isa(obj,'simpleModel'));
            nbClass = length(obj.labs);
            k = 0;
            if nargin==1
                for i=1:nbClass
                    for j=i+1:nbClass
                        k=k+1;
                        ind = sort([find(obj.SD.trainlab==obj.labs(i));find(obj.SD.trainlab==obj.labs(j))]);
                        tl = obj.SD.trainlab(ind);
                        tl(tl==obj.labs(i))=-1;
                        tl(tl==obj.labs(j))=1;
                        if strcmp(obj.SD.kernelType,'precomputed')
                            tt = obj.SD.kernelCache(ind,ind);
                        else
                            tt = obj.SD.trainvec(:,ind);
                        end
                        SDt{k} = simpleData(tt,tl);
                        SDt{k}.kernelParam = obj.SD.kernelParam;
                        SDt{k}.kernelType = obj.SD.kernelType;
                        SMt{k} = feval(class(obj),SDt{k},obj.Slacks(ind));
                        SMt{k}.train();
                    end
                end
            elseif nargin==2
                assert(length(SMt)==(nbClass-1)*nbClass/2);
                for i=1:nbClass
                    for j=i+1:nbClass
                        k=k+1;
                        if (sum(SMt{k}.SD.kernelParam-obj.SD.kernelParam)==0|| not(strcmp(SMt{k}.SD.kernelType,obj.SD.kernelType)))
                            SMt{k}.SD.clear;
                            SMt{k}.SD.kernelType = obj.SD.kernelType;
                            SMt{k}.SD.kernelParam = obj.SD.kernelParam;
                        end
                        ind = sort([find(obj.SD.trainlab==obj.labs(i));find(obj.SD.trainlab==obj.labs(j))]);
                        
                        SMt{k} = feval(class(obj),SMt{k}.SD,obj.Slacks(ind),SMt{k}.Il,SMt{k}.Iu,SMt{k}.Iw);
                        SMt{k}.train();
                    end
                end
            end
        end
        
        
        function [ypred,perf] = test1vs1(obj,SMt,testvec,testlab)
            assert(isa(obj,'simpleModel'));
            nbClass = length(obj.labs);
            k = 0;
            vote = zeros(size(testvec,2),(nbClass*(nbClass-1))/2);
            for i=1:nbClass
                for j=i+1:nbClass
                    k=k+1;
                    if not(strcmp(obj.SD.kernelType,'precomputed'))
                        vote(:,k) = SMt{k}.test(testvec);
                    else
                        ind = sort([find(obj.SD.trainlab==obj.labs(i));find(obj.SD.trainlab==obj.labs(j))]);
                        vote(:,k) = SMt{k}.test(testvec(ind,:));
                    end
                    ind1 = vote(:,k)<0;
                    ind2 = vote(:,k)>=0;
                    vote(ind1,k) = i;
                    vote(ind2,k) = j;
                end
            end
            ypred = mode(vote,2);
            if nargin==4
                perf = 100*length(find(ypred==testlab))/length(testlab);
            else
                perf = -1;
            end
            
        end
        
        
        
        function SMt = train1vsAll(obj,SMt)
            assert(isa(obj,'simpleModel'));
            nbClass = length(obj.labs);
            if nargin==1
                for i=1:nbClass
                    tl = obj.SD.trainlab;
                    tl(tl~=obj.labs(i))=-1;
                    tl(tl==obj.labs(i))=1;
                    if isempty(obj.SD.trainvec)
                        SDt{i} = simpleData(obj.SD.kernelCache,tl);
                    else
                        SDt{i} = simpleData(obj.SD.trainvec,tl);
                    end
                    SDt{i}.kernelParam = obj.SD.kernelParam;
                    SDt{i}.kernelType = obj.SD.kernelType;
                    SMt{i} = feval(class(obj),SDt{i},obj.Slacks);
                    SMt{i}.train();
                end
            elseif nargin==2
                assert(length(SMt)==nbClass);
                for i=1:nbClass
                    if (sum(SMt{i}.SD.kernelParam-obj.SD.kernelParam)==0|| not(strcmp(SMt{i}.SD.kernelType,obj.SD.kernelType)))
                        SMt{i}.SD.kernelType = obj.SD.kernelType;
                        SMt{i}.SD.kernelParam = obj.SD.kernelParam;
                    end
                    SMt{i}.SD.clear;
                        
                    SMt{i} = feval(class(obj),SMt{i}.SD,obj.Slacks,SMt{i}.Il,SMt{i}.Iu,SMt{i}.Iw);
                    SMt{i}.train();
                end
            end
        end
        
        
        function [ypred,perf] = test1vsAll(obj,SMt,testvec,testlab)
            assert(isa(obj,'simpleModel'));
            nbClass = length(obj.labs);
            vote = zeros(size(testvec,2),nbClass);
            for i=1:nbClass
                vote(:,i) = SMt{i}.test(testvec);
            end
            [~,ypred] = max(vote,[],2);
            if nargin==4
                perf = 100*length(find(ypred==testlab))/length(testlab);
            else
                perf = -1;
            end
            
        end
        
        
    end
end
