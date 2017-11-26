function f = kernelTest(noyau,kernelParam,vect, labs, test,testlabs, Q, lambda)


switch noyau
    case 'rbf'
        unitvec = ones(size(test,2),1);
        dot_a = sum(test.*test,1);
        dot_b = sum(vect.*vect,1);
        f = test' * vect;
        if nargin==8
            if lambda==0
                for i=1:length(labs)
                    f(:,i) = (labs(i)==testlabs) .* ...
                        (exp(kernelParam * (2 * f(:,i) - dot_a' - dot_b(i) * unitvec)));
                end
            else
                A = lambda/(1+Q*lambda);
                B = 1/lambda + (Q-1);
                for i=1:length(labs)
                    f(:,i) = (1-(B+1)*(labs(i)==testlabs)).* ...
                        (-A*exp(kernelParam * (2 * f(:,i) - dot_a' - dot_b(i) * unitvec)));
                end
            end
        else
            for i=1:length(labs)
                f(:,i) = labs(i)*exp(kernelParam * (2 * f(:,i) - dot_a' - dot_b(i) * unitvec)).*testlabs;
            end
        end
    case 'poly'
        f = (test' * vect + kernelParam(2)).^kernelParam(1);
        if nargin==8
            A= lambda/(1+Q*lambda);
            if lambda==0
                for i=1:length(labs)
                    f(:,i) = labs(i) * (f(:,i) .* testlabs);
                end
            else
                B= 1/lambda + (Q-1);
                for i=1:length(labs)
                    f(:,i) = (-B).^(labs(i)==testlabs).* (-A*f(:,i));
                end
            end
        else
            for i=1:length(labs)
                f(:,i) = labs(i) * (f(:,i) .* testlabs);
            end
            
        end
    case 'gauss'
        unitvec = ones(size(test,2),1);
        dot_a = sum(test.*test,1);
        dot_b = sum(vect.*vect,1);
        f = (1/sqrt(2*pi*noyau.sigma))*test' * vect;
        if nargin==8
            if lambda==0
                for i=1:length(labs)
                    f(:,i) = (labs(i)==testlabs) .* ...
                        (exp(noyau.sigma * (2 * f(:,i) - dot_a' - dot_b(i) * unitvec)));
                end
            else
                A = lambda/(1+Q*lambda);
                B = 1/lambda + (Q-1);
                for i=1:length(labs)
                    f(:,i) = (1-(B-1)*(labs(i)==testlabs)).* ...
                        (-A*exp(noyau.sigma * (2 * f(:,i) - dot_a' - dot_b(i) * unitvec)));
                end
            end
        else
            for i=1:length(labs)
                f(:,i) = labs(i)*exp(noyau.sigma * (2 * f(:,i) - dot_a' - dot_b(i) * unitvec)).*testlabs;
            end
        end
    case 'ratio'
        
        unitvec = ones(size(test,2),1);
        dot_a = sum(test.*test,1);
        dot_b = sum(vect.*vect,1);
        f = test' * vect;
        if nargin==8
            if lambda==0
                for i=1:length(labs)
                    f(:,i) = (labs(i)==testlabs) .* ...
                        (1-((-2 * f(:,i) + dot_a' + dot_b(i) * unitvec)./...
                        (-2 * f(:,i) + dot_a' + dot_b(i) * unitvec+noyau.sigma)));
                end
            else
                A = lambda/(1+Q*lambda);
                B = 1/lambda + (Q-1);
                for i=1:length(labs)
                    f(:,i) = (1-(B-1)*(labs(i)==testlabs)).* ...
                        (-A*(1-((-2 * f(:,i) + dot_a' + dot_b(i) * unitvec)./...
                        (-2 * f(:,i) + dot_a' + dot_b(i) * unitvec+noyau.sigma))) );
                end
            end
        else
            for i=1:length(labs)
                f(:,i) = labs(i)*((1-((-2 * f(:,i) + dot_a' + dot_b(i) * unitvec)./...
                    (-2 * f(:,i) + dot_a' + dot_b(i) * unitvec+noyau.sigma))) .*testlabs);
            end
        end
    case 'tanh'
        f = tanh(kernelParam(1) * test' * vect + kernelParam(2));
        if nargin==8
            A= lambda/(1+Q*lambda);
            if lambda==0
                for i=1:length(labs)
                    f(:,i) = labs(i) * (f(:,i) .* testlabs);
                end
            else
                B= 1/lambda + (Q-1);
                for i=1:length(labs)
                    f(:,i) = (-B).^(labs(i)==testlabs).* (-A*f(:,i));
                end
            end
        else
            for i=1:length(labs)
                f(:,i) = labs(i) * (f(:,i) .* testlabs);
            end
            
        end
        
        
    case 'precomputed'
        f = test;
    case 'graph'
        f = zeros(length(testlabs),length(labs));
        if nargin==8
            A= lambda/(1+Q*lambda);
            if lambda==0
                for i=1:length(labs)
                    for j=1:length(test)
                        nbetiqnoeuds=size(vect(i).etiquetteNoeud,2);
                        nbetiqarcs=size(vect(i).edgeLabel(1,:),2);
                        sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                        sigma(1,1:nbetiqnoeuds) = kernelParam(1);
                        sigma(1,nbetiqnoeuds+1:end)=kernelParam(2);
                        f(j,i) =(labs(i)==testlabs) .* ...
                            (-A*Kchemin(vect(i),test(j),sigma,3));
                    end
                end
            else
                B = 1/lambda + (Q-1);
                for i=1:length(labs)
                    for j=1:length(test)
                        nbetiqnoeuds=size(vect(i).etiquetteNoeud,2);
                        nbetiqarcs=size(vect(i).edgeLabel(1,:),2);
                        sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                        sigma(1,1:nbetiqnoeuds) = kernelParam(1);
                        sigma(1,nbetiqnoeuds+1:end)=kernelParam(2);
                        f(j,i) =(1-(B-1)*(labs(i)==testlabs)) .* ...
                            (-A*Kchemin(vect(i),test(j),sigma,kernelParam(3)));
                    end
                end
            end
        else
            for i=1:length(labs)
                for j=1:length(testlabs)
                    nbetiqnoeuds=size(vect(i).etiquetteNoeud,2);
                    nbetiqarcs=size(vect(i).edgeLabel(1,:),2);
                    sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                    sigma(1,1:nbetiqnoeuds) = kernelParam(1);
                    sigma(1,nbetiqnoeuds+1:end)=kernelParam(2);
                    f(i,j) = labs(i)*testlabs(j)*Kchemin(vect(i),test(j),sigma,kernelParam(3));
                end
            end
            
        end
    case 'graphMax'
        f = zeros(length(testlabs),length(labs));
        if nargin==8
            A= lambda/(1+Q*lambda);
            if lambda==0
                for i=1:length(labs)
                    for j=1:length(test)
                        nbetiqnoeuds=size(vect(i).etiquetteNoeud,2);
                        nbetiqarcs=size(vect(i).edgeLabel(1,:),2);
                        sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                        sigma(1,1:nbetiqnoeuds) = kernelParam(1);
                        sigma(1,nbetiqnoeuds+1:end)=kernelParam(2);
                        f(j,i) =(labs(i)==testlabs) .* ...
                            (-A*KcheminMax(vect(i),test(j),sigma,kernelParam(3)));
                    end
                end
            else
                B = 1/lambda + (Q-1);
                for i=1:length(labs)
                    for j=1:length(test)
                        nbetiqnoeuds=size(vect(i).etiquetteNoeud,2);
                        nbetiqarcs=size(vect(i).edgeLabel(1,:),2);
                        sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                        sigma(1,1:nbetiqnoeuds) = kernelParam(1);
                        sigma(1,nbetiqnoeuds+1:end)=kernelParam(2);
                        f(j,i) =(1-(B-1)*(labs(i)==testlabs)) .* ...
                            (-A*KcheminMax(vect(i),test(j),sigma,kernelParam(3)));
                    end
                end
            end
        else
            for i=1:length(labs)
                for j=1:length(testlabs)
                    nbetiqnoeuds=size(vect(i).etiquetteNoeud,2);
                    nbetiqarcs=size(vect(i).edgeLabel(1,:),2);
                    sigma=ones(1,nbetiqnoeuds+nbetiqarcs); % general case
                    sigma(1,1:nbetiqnoeuds) = kernelParam(1);
                    sigma(1,nbetiqnoeuds+1:end)=kernelParam(2);
                    f(i,j) = labs(i)*testlabs(j)*Kchemin(vect(i),test(j),sigma,3);
                end
            end
            
        end
end


