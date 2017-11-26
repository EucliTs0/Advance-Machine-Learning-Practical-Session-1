classdef simpleData < handle
    % SIMPLEDATA This class contains all data that are used in simpleModel
    % solvers. It stores initial data (or a pre-computed kernel), computes
    % the kernel matrix on-demand with a (simple) cache system.
    % See also : simpleModel
    
    % Create by Gaëlle Loosli
    % 16/02/2011
    % Last Modified 16/02/2011 by G.L.
    
    
    properties
        trainvec; % stored as column individuals
        trainlab; % stored as column
        kernelType = 'rbf';
        kernelParam = 1;
        kernelCache; % computed on the fly or pre-computed
        kernelCachePur; % computed on the fly or pre-computed
        n;
    end
    
    properties (SetAccess = private)
        indx; % cached rows
        relIndx; % position of cached rows
        premult;
    end
    
    
    methods
        % contructor
        function SD = simpleData(tt,tl)
            assert(nargin==2,'simpleData:wrongInput','SimpleData needs training points or pre-computed kernel matrix and associated labels : two arguments');
            assert(size(tl,1)==size(tt,2) && size(tl,2)==1,'simpleData:wrongInput','Labels should be in a column vector, training point stored as columns');
            if size(tt,1)~=size(tt,2)
                SD.trainvec = tt;
                SD.trainlab = tl;
                SD.relIndx = zeros(1,size(tt,2));
                SD.n = size(tt,2);
                if not(isstruct(tt))
                    SD.premult = tt.*tt;
                end
                SD.kernelCache = zeros(SD.n,min(500,SD.n));
                SD.kernelCachePur = zeros(SD.n,min(500,SD.n));
            else
                SD.kernelCache = tt.*(tl*tl');
                SD.kernelCachePur = SD.kernelCache;
                SD.trainvec = tt;
                SD.trainlab = tl;
                SD.n = size(tt,1);
                SD.indx = 1:size(tt,1);
                SD.relIndx = 1:size(tt,1);
                SD.kernelType='precomputed';
            end
        end
        
        function set.kernelType(obj,type)
            obj.kernelType = type;
        end
        
        function type = get.kernelType(obj)
            type = obj.kernelType;
        end
        
        function  set.kernelParam(obj,param)
            obj.kernelParam = param;
        end
        
        function param = get.kernelParam(obj)
            param = obj.kernelParam;
        end
        
        
        
        function indd = getKernelValues(obj,indx,indy)
            %a = setdiff(indx, obj.indx);
            a = indx(~(ismember(indx,obj.indx)));
            if ~isempty(a)
                obj.computeKernel(a);
                obj.updateCache();
            end
            %K = obj.kernelCache(indy,obj.relIndx(indx));
            indd = obj.relIndx(indx);
        end
        
        function obj = computeKernel(obj,ind)
            relInd = length(obj.indx)+1:length(obj.indx)+length(ind);
            kernelLab(obj,ind,relInd);
            obj.relIndx(ind) = relInd;
            obj.indx = [obj.indx,ind];
        end
        
        
        function indd = getDocKernelValues(obj,indx,indy,Q,lambda)
            %a = setdiff(indx, obj.indx);
            a = indx(~(ismember(indx,obj.indx)));
            if ~isempty(a)
                obj.computeDocKernel(a,Q,lambda);
                obj.updateCache();
            end
            %K = obj.kernelCache(indy,obj.relIndx(indx));
            indd = obj.relIndx(indx);
        end
        
        
        function indd = getDoc2KernelValues(obj,indx,indy,Q,lambda,A,Z)
            %a = setdiff(indx, obj.indx);
            a = indx(~(ismember(indx,obj.indx)));
            if ~isempty(a)
                obj.computeDoc2Kernel(a,Q,lambda,A,Z);
                obj.updateCache();
            end
            %K = obj.kernelCache(indy,obj.relIndx(indx));
            indd = obj.relIndx(indx);
        end
        
        function obj = computeDocKernel(obj,ind,Q,lambda)
            relInd = length(obj.indx)+1:length(obj.indx)+length(ind);
            obj.kernelDocLab(ind,Q,lambda,relInd);
            obj.relIndx(ind) = relInd;
            obj.indx = [obj.indx,ind];
        end
        
        
        function obj = computeDoc2Kernel(obj,ind,Q,lambda,A,Z)
            relInd = length(obj.indx)+1:length(obj.indx)+length(ind);
            obj.kernelDoc2Lab(ind,Q,lambda,A,Z,relInd);
            obj.relIndx(ind) = relInd;
            obj.indx = [obj.indx,ind];
        end
        
        function obj = kernelLab(obj,ind,relInd)
            if strcmp(obj.kernelType,'rbf')
                dot_b = sum(obj.premult(:,ind),1);
                dot_a = sum(obj.premult,1)';
                unitvec = ones(obj.n,1);
                obj.kernelCache(:,relInd) =   obj.trainvec'*obj.trainvec(:,ind);
                for i=1:length(ind)
                    a = obj.kernelParam(1) * ...
                        (2 * obj.kernelCache(:,relInd(i)) - dot_a - dot_b(i) * unitvec);
                    obj.kernelCache(:,relInd(i)) = obj.trainlab(ind(i)) * (exp(a) .* obj.trainlab);
                end
            end
            
            if strcmp(obj.kernelType,'rbfInv')
                dot_b = sum(obj.premult(:,ind),1);
                dot_a = sum(obj.premult,1)';
                unitvec = ones(obj.n,1);
                obj.kernelCache(:,relInd) =   obj.trainvec'*obj.trainvec(:,ind);
                kC(:,relInd) = -obj.trainvec'*obj.trainvec(:,ind);
                for i=1:length(ind)
                    a = obj.kernelParam(1) * ...
                        (2 * obj.kernelCache(:,relInd(i)) - dot_a - dot_b(i) * unitvec);
                    obj.kernelCache(:,relInd(i)) = obj.trainlab(ind(i)) * (exp(a) .* obj.trainlab);
                    b = obj.kernelParam(1) * ...
                        (2 * obj.kernelCache(:,relInd(i)) - dot_a + dot_b(i) * unitvec);
                    kC(:,relInd(i)) =  obj.trainlab(ind(i)) * (exp(a) .* obj.trainlab);
                    obj.kernelCache(:,relInd(i)) = max(obj.kernelCache(:,relInd(i)),kC(:,relInd(i)));
                    keyboard
                end
            end
            
            if strcmp(obj.kernelType,'poly')
                assert(length(obj.kernelParam)==2);
                obj.kernelCache(:,relInd) = (obj.trainvec' * obj.trainvec(:,ind) +...
                    obj.kernelParam(2)).^obj.kernelParam(1);
                for i=1:length(ind)
                    obj.kernelCache(:,relInd(i)) = obj.trainlab(ind(i)) * (obj.kernelCache(:,relInd(i)) .* obj.trainlab);
                end
            end
            
            
            if strcmp(obj.kernelType,'tanh')
                assert(length(obj.kernelParam)==2);
                obj.kernelCache(:,relInd) = tanh(obj.kernelParam(1) * obj.trainvec' * obj.trainvec(:,ind) +...
                    obj.kernelParam(2));
                for i=1:length(ind)
                    obj.kernelCache(:,relInd(i)) = obj.trainlab(ind(i)) * (obj.kernelCache(:,relInd(i)) .* obj.trainlab);
                end
            end
            
            
            if strcmp(obj.kernelType,'precomputed')
                obj.kernelCache = obj.trainvec.*(obj.trainlab*obj.trainlab');
            end
            
            if strcmp(obj.kernelType,'graph')
                assert(length(obj.kernelParam)==3);
                for i=1:length(ind)
                    for j=1:length(obj.trainlab)
                        nbetiqnoeuds=size(obj.trainvec(relInd(i)).etiquetteNoeud,2);
                        nbetiqarcs=size(obj.trainvec(relInd(i)).edgeLabel(1,:),2);
                        sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                        sigma(1,1:nbetiqnoeuds) = obj.kernelParam(1);
                        sigma(1,nbetiqnoeuds+1:end)=obj.kernelParam(2);
                        obj.kernelCache(i,relInd(j)) = obj.trainlab(ind(i))*obj.trainlab(relInd(j))*Kchemin(obj.trainvec(ind(i)),obj.trainvec(j),sigma,obj.kernelParam(3));
                    end
                end
            end
            
            
            if strcmp(obj.kernelType,'graphMax')
                assert(length(obj.kernelParam)==3);
                for i=1:length(ind)
                    for j=1:length(obj.trainlab)
                        nbetiqnoeuds=size(obj.trainvec(relInd(i)).etiquetteNoeud,2);
                        nbetiqarcs=size(obj.trainvec(relInd(i)).edgeLabel(1,:),2);
                        sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                        sigma(1,1:nbetiqnoeuds) = obj.kernelParam(1);
                        sigma(1,nbetiqnoeuds+1:end)=obj.kernelParam(2);
                        obj.kernelCache(i,relInd(j)) = obj.trainlab(ind(i))*obj.trainlab(relInd(j))*KcheminMax(obj.trainvec(ind(i)),obj.trainvec(j),sigma,obj.kernelParam(3));
                    end
                end
            end
        end
        
        function obj = kernelDocLab(obj,ind,Q,lambda,relInd)
            if strcmp(obj.kernelType,'rbf')
                assert(length(obj.kernelParam)==1);
                dot_b = sum(obj.premult(:,ind),1);
                dot_a = sum(obj.premult,1)';
                unitvec = ones(obj.n,1);
                A = lambda/(1+Q*lambda);
                obj.kernelCache(:,relInd) =   obj.trainvec'*obj.trainvec(:,ind);
                if lambda==0
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) =(obj.trainlab(ind(i))==obj.trainlab) .* (exp(obj.kernelParam(1) * ...
                            (2 * obj.kernelCache(:,relInd(i)) - dot_a - dot_b(i) * unitvec)));
                    end
                else
                    B = 1/lambda + (Q-1);
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) =(1-(B+1)*(obj.trainlab(ind(i))==obj.trainlab)) .* (-A*exp(obj.kernelParam(1) * ...
                            (2 * obj.kernelCache(:,relInd(i)) - dot_a - dot_b(i) * unitvec)));
                    end
                end
            end
            
            if strcmp(obj.kernelType,'poly')
                assert(length(obj.kernelParam)==2);
                obj.kernelCache(:,relInd) = (obj.trainvec' * obj.trainvec(:,ind) +...
                    obj.kernelParam(2)).^obj.kernelParam(1);
                A= lambda/(1+Q*lambda);
                if lambda==0
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) = obj.trainlab(ind(i)) * (obj.kernelCache(:,relInd(i)) .* obj.trainlab);
                    end
                else
                    B= 1/lambda + (Q-1);
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) = -B.^(obj.trainlab(ind(i))==obj.trainlab).*(-A*obj.kernelCache(:,relInd(i)));
                    end
                end
                
            end
            
            if strcmp(obj.kernelType,'tanh')
                assert(length(obj.kernelParam)==2);
                obj.kernelCache(:,relInd) = tanh(obj.kernelParam(1) * obj.trainvec' * obj.trainvec(:,ind) +...
                    obj.kernelParam(2));
                A= lambda/(1+Q*lambda);
                if lambda==0
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) = obj.trainlab(ind(i)) * (obj.kernelCache(:,relInd(i)) .* obj.trainlab);
                    end
                else
                    B= 1/lambda + (Q-1);
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) = -B.^(obj.trainlab(ind(i))==obj.trainlab).*(-A*obj.kernelCache(:,relInd(i)));
                    end
                end
                
            end
            
            if strcmp(obj.kernelType,'precomputed')
                A= lambda/(1+Q*lambda);
                if lambda==0
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) = obj.trainlab(ind(i)) * (obj.kernelCache(:,relInd(i)) .* obj.trainlab);
                    end
                else
                    B= 1/lambda + (Q-1);
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) = -B.^(obj.trainlab(ind(i))==obj.trainlab).*(-A*obj.kernelCache(:,relInd(i)));
                    end
                end
            end
            
            
            
            
            if strcmp(obj.kernelType,'graph')
                assert(length(obj.kernelParam)==3);
                A= lambda/(1+Q*lambda);
                if lambda==0
                    for i=1:length(ind)
                        for j=1:length(obj.trainlab)
                            nbetiqnoeuds=size(obj.trainvec(i).etiquetteNoeud,2);
                            nbetiqarcs=size(obj.trainvec(i).edgeLabel(1,:),2);
                            sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                            sigma(1,1:nbetiqnoeuds) = obj.kernelParam(1);
                            sigma(1,nbetiqnoeuds+1:end)=obj.kernelParam(2);
                            obj.kernelCache(j,relInd(i)) =(obj.trainlab(ind(i))==obj.trainlab(j)) .* ...
                                (-A*Kchemin(obj.trainvec(ind(i)),obj.trainvec(j),sigma,obj.kernelParam(3)));
                        end
                    end
                else
                    B = 1/lambda + (Q-1);
                    for i=1:length(ind)
                        for j=1:length(obj.trainlab)
                            nbetiqnoeuds=size(obj.trainvec(relInd(i)).etiquetteNoeud,2);
                            nbetiqarcs=size(obj.trainvec(relInd(i)).edgeLabel(1,:),2);
                            sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                            sigma(1,1:nbetiqnoeuds) = obj.kernelParam(1);
                            sigma(1,nbetiqnoeuds+1:end)=obj.kernelParam(2);
                            obj.kernelCache(j,relInd(i)) =(1-(B-1)*(obj.trainlab(ind(i))==obj.trainlab(j))) .* ...
                                (-A*Kchemin(obj.trainvec(ind(i)),obj.trainvec(j),sigma,obj.kernelParam(3)));
                        end
                    end
                end
                
            end
            if strcmp(obj.kernelType,'graphMax')
                assert(length(obj.kernelParam)==3);
                A= lambda/(1+Q*lambda);
                if lambda==0
                    for i=1:length(ind)
                        for j=1:length(obj.trainlab)
                            nbetiqnoeuds=size(obj.trainvec(i).etiquetteNoeud,2);
                            nbetiqarcs=size(obj.trainvec(i).edgeLabel(1,:),2);
                            sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                            sigma(1,1:nbetiqnoeuds) = obj.kernelParam(1);
                            sigma(1,nbetiqnoeuds+1:end)=obj.kernelParam(2);
                            obj.kernelCache(j,relInd(i)) =(obj.trainlab(ind(i))==obj.trainlab(j)) .* ...
                                (-A*KcheminMax(obj.trainvec(ind(i)),obj.trainvec(j),sigma,obj.kernelParam(3)));
                        end
                    end
                else
                    B = 1/lambda + (Q-1);
                    for i=1:length(ind)
                        for j=1:length(obj.trainlab)
                            nbetiqnoeuds=size(obj.trainvec(relInd(i)).etiquetteNoeud,2);
                            nbetiqarcs=size(obj.trainvec(relInd(i)).edgeLabel(1,:),2);
                            sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                            sigma(1,1:nbetiqnoeuds) = obj.kernelParam(1);
                            sigma(1,nbetiqnoeuds+1:end)=obj.kernelParam(2);
                            obj.kernelCache(j,relInd(i)) =(1-(B-1)*(obj.trainlab(ind(i))==obj.trainlab(j))) .* ...
                                (-A*KcheminMax(obj.trainvec(ind(i)),obj.trainvec(j),sigma,obj.kernelParam(3)));
                        end
                    end
                end
                
            end
        end
        
        function obj = kernelDoc2Lab(obj,ind,Q,lambda,A,Z,relInd)
            
            gsame = ((A^2+(Q-1)*lambda^2)/Z^2);
            gdiff = ((-2*A*lambda+ (Q-2)*(lambda^2))/Z^2);
            
            if strcmp(obj.kernelType,'rbf')
                assert(length(obj.kernelParam)==1);
                dot_b = sum(obj.premult(:,ind),1);
                dot_a = sum(obj.premult,1)';
                unitvec = ones(obj.n,1);
                %A = lambda/(1+Q*lambda);
                obj.kernelCachePur(:,relInd) =   obj.trainvec'*obj.trainvec(:,ind);
                for i=1:length(ind)
                    obj.kernelCachePur(:,relInd(i)) =(exp(obj.kernelParam(1) * ...
                        (2 * obj.kernelCachePur(:,relInd(i)) - dot_a - dot_b(i) * unitvec)));
                    
                    obj.kernelCache(:,relInd(i)) =(gsame*(obj.trainlab(ind(i))==obj.trainlab)+...
                        gdiff*(obj.trainlab(ind(i))~=obj.trainlab)) .*obj.kernelCachePur(:,relInd(i));
                end
            end
            
            if strcmp(obj.kernelType,'poly')
                assert(length(obj.kernelParam)==2);
                obj.kernelCache(:,relInd) = (obj.trainvec' * obj.trainvec(:,ind) +...
                    obj.kernelParam(2)).^obj.kernelParam(1);
                A= lambda/(1+Q*lambda);
                if lambda==0
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) = obj.trainlab(ind(i)) * (obj.kernelCache(:,relInd(i)) .* obj.trainlab);
                    end
                else
                    B= 1/lambda + (Q-1);
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) = -B.^(obj.trainlab(ind(i))==obj.trainlab).*(-A*obj.kernelCache(:,relInd(i)));
                    end
                end
                
            end
            
            if strcmp(obj.kernelType,'tanh')
                assert(length(obj.kernelParam)==2);
                obj.kernelCache(:,relInd) = tanh(obj.kernelParam(1) * obj.trainvec' * obj.trainvec(:,ind) +...
                    obj.kernelParam(2));
                A= lambda/(1+Q*lambda);
                if lambda==0
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) = obj.trainlab(ind(i)) * (obj.kernelCache(:,relInd(i)) .* obj.trainlab);
                    end
                else
                    B= 1/lambda + (Q-1);
                    for i=1:length(ind)
                        obj.kernelCache(:,relInd(i)) = -B.^(obj.trainlab(ind(i))==obj.trainlab).*(-A*obj.kernelCache(:,relInd(i)));
                    end
                end
                
            end
            
            if strcmp(obj.kernelType,'precomputed')
                for i=1:length(ind)
                    
                    obj.kernelCache(:,relInd(i)) =(gsame*(obj.trainlab(ind(i))==obj.trainlab)+...
                        gdiff*(obj.trainlab(ind(i))~=obj.trainlab)) .*obj.kernelCachePur(:,relInd(i));
                end
            end
            
            
            
            
            if strcmp(obj.kernelType,'graph')
                assert(length(obj.kernelParam)==3);
                A= lambda/(1+Q*lambda);
                if lambda==0
                    for i=1:length(ind)
                        for j=1:length(obj.trainlab)
                            nbetiqnoeuds=size(obj.trainvec(i).etiquetteNoeud,2);
                            nbetiqarcs=size(obj.trainvec(i).edgeLabel(1,:),2);
                            sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                            sigma(1,1:nbetiqnoeuds) = obj.kernelParam(1);
                            sigma(1,nbetiqnoeuds+1:end)=obj.kernelParam(2);
                            obj.kernelCache(j,relInd(i)) =(obj.trainlab(ind(i))==obj.trainlab(j)) .* ...
                                (-A*Kchemin(obj.trainvec(ind(i)),obj.trainvec(j),sigma,obj.kernelParam(3)));
                        end
                    end
                else
                    B = 1/lambda + (Q-1);
                    for i=1:length(ind)
                        for j=1:length(obj.trainlab)
                            nbetiqnoeuds=size(obj.trainvec(relInd(i)).etiquetteNoeud,2);
                            nbetiqarcs=size(obj.trainvec(relInd(i)).edgeLabel(1,:),2);
                            sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                            sigma(1,1:nbetiqnoeuds) = obj.kernelParam(1);
                            sigma(1,nbetiqnoeuds+1:end)=obj.kernelParam(2);
                            obj.kernelCache(j,relInd(i)) =(1-(B-1)*(obj.trainlab(ind(i))==obj.trainlab(j))) .* ...
                                (-A*Kchemin(obj.trainvec(ind(i)),obj.trainvec(j),sigma,obj.kernelParam(3)));
                        end
                    end
                end
                
            end
            if strcmp(obj.kernelType,'graphMax')
                assert(length(obj.kernelParam)==3);
                A= lambda/(1+Q*lambda);
                if lambda==0
                    for i=1:length(ind)
                        for j=1:length(obj.trainlab)
                            nbetiqnoeuds=size(obj.trainvec(i).etiquetteNoeud,2);
                            nbetiqarcs=size(obj.trainvec(i).edgeLabel(1,:),2);
                            sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                            sigma(1,1:nbetiqnoeuds) = obj.kernelParam(1);
                            sigma(1,nbetiqnoeuds+1:end)=obj.kernelParam(2);
                            obj.kernelCache(j,relInd(i)) =(obj.trainlab(ind(i))==obj.trainlab(j)) .* ...
                                (-A*KcheminMax(obj.trainvec(ind(i)),obj.trainvec(j),sigma,obj.kernelParam(3)));
                        end
                    end
                else
                    B = 1/lambda + (Q-1);
                    for i=1:length(ind)
                        for j=1:length(obj.trainlab)
                            nbetiqnoeuds=size(obj.trainvec(relInd(i)).etiquetteNoeud,2);
                            nbetiqarcs=size(obj.trainvec(relInd(i)).edgeLabel(1,:),2);
                            sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                            sigma(1,1:nbetiqnoeuds) = obj.kernelParam(1);
                            sigma(1,nbetiqnoeuds+1:end)=obj.kernelParam(2);
                            obj.kernelCache(j,relInd(i)) =(1-(B-1)*(obj.trainlab(ind(i))==obj.trainlab(j))) .* ...
                                (-A*KcheminMax(obj.trainvec(ind(i)),obj.trainvec(j),sigma,obj.kernelParam(3)));
                        end
                    end
                end
                
            end
        end
        
        
        
        function updateCache(obj)
            if length(obj.indx)+10>size(obj.kernelCache,2) && not(strcmp(obj.kernelType,'precomputed'))
                obj.kernelCache = [obj.kernelCache,zeros(obj.n,100)];
            end
        end
        
        
        function clear(obj)
            obj.indx = [];
            obj.relIndx = obj.relIndx*0;
            if strcmp(obj.kernelType,'precomputed')
                obj.indx = 1:size(obj.trainvec,1);
                obj.relIndx = 1:size(obj.trainvec,1);
                if (max(obj.trainlab)==1)
                    obj.kernelCache = obj.trainvec.*(obj.trainlab*obj.trainlab');
                end
            else
                obj.kernelCache = zeros(obj.n,500);
            end
        end
        
    end
    
end

