% test
clear all;
n=3000;
d=2;
napp=90;
nval=300;

global SMscale;
global SMstep;
SMscale =1;
SMstep = 0.05;
  
    x = rand(d,n);
    y = -ones(n,1);
    points = abs(0.2*randn(d,5)+0.4);
    for i=1:size(points,2)
        m = diag((x-points(:,i)*ones(1,n))'*(x-points(:,i)*ones(1,n)));
        seuil = 0.04*randn(1);
        y(m<abs(seuil))=1;
    end
      
    
    xa = x(:,find(y==1,napp));
    ya = ones(napp,1);
    xv = [x(:,find(y==1,napp+nval))];
    xv(:,1:napp) = [];
    yv = [ones(nval,1)];
    xt = x;
    yt = y;
    
     nu = 0.1; 
    %nu = [0.05 0.1 0.2];
    %ker = [0.1 0.5 1 2 5 10 20 50 100];
    ker = [0.1 1 10 100];
    %ker2 = [[4 1];[3 1];[2 1];[1 1];];
    %ker2 = [[8 1];[24 1];];
    
    perf = zeros(6,length(nu));
    kerV = zeros(2,length(nu));
    ker2V = zeros(2,length(nu),2);
    
     
        for p = 1:length(nu)
            %for k = 1:size(ker2, 1)
            for k=1:length(ker)
                SD2 = simpleData(xa,ya);
                %SD2.kernelType = 'poly';
                SD2.kernelType = 'rbf';
                SD2.kernelParam = ker(k);
                %SD2.kernelParam = ker2(k, :);


                SM3 = simpleModelSVDD(SD2,1/(nu(p)*length(ya)));
                SM3.train;
                
                %figure(1)
                subplot(2,2,k)
                SM3.print2D();
                plot(xt(1,yt==1),xt(2,yt==1),'r.','markersize',4);
                plot(xt(1,yt==-1),xt(2,yt==-1),'g.','markersize',4);
                %hold off;
                %title('OC-SVDD - Polynomial kernel of degree 32 and 16');
                axis([0 1 0 1]);
                drawnow;
            end
        end
        %%
%                 [~,perfV] = SM3.test(xv,yv);
%                 if (perfV>perf(5,p))
%                     perf(5,p) = perfV;
%                     kerV(1,p) = k;
%                 end



        [bestP,indb] = max(perf,[],2);

        global SMscale;
        global SMstep;
        SMscale =1;
        SMstep = 0.05;


            SD2 = simpleData(xa,ya);
            SD2.kernelType = 'poly';
            SD2.kernelParam = ker2(indb(5),:);
            SM3 = simpleModelSVDD(SD2,1/(nu(indb(5))*length(ya)));
            %SD2.kernelType = 'rbf';
            %SD2.kernelParam = ker(indb(5));
            %SM3 = simpleModelSVDD(SD2,1/(nu(indb(5))*length(ya)));
            SM3.train;
            %[~,perfTest(5,loop)] = SM3.test(xt,yt);
            figure(1)
            SM3.print2D();
            plot(xt(1,yt==1),xt(2,yt==1),'r.','markersize',4);
            plot(xt(1,yt==-1),xt(2,yt==-1),'g.','markersize',4);
            hold off;
            title('SVDD');
            axis([0 1 0 1]);

    %         SD2 = simpleData(xa,ya);
    %         SD2.kernelType = 'poly';
    %         SD2.kernelParam = ker2(indb(5),:);
    %         SM3 = simpleModelSVDD(SD2,1/(nu(indb(5))*length(ya)));
    %         SM3.train;
    %         [~,perfTest(5,loop)] = SM3.test(xt,yt);
    %         subplot(3,2,5);
    %         SM3.print2D();
    %         plot(xt(1,yt==1),xt(2,yt==1),'r.','markersize',4);
    %         plot(xt(1,yt==-1),xt(2,yt==-1),'g.','markersize',4);
    %         hold off;
    %         title('SVDD');
    %         axis([0 1 0 1]);




        %[SM.dm' SM2.dm']
        %[perfTest(1,loop) perfTest(2,loop) perfTest(3,loop) perfTest(4,loop) perfTest(5,loop) perfTest(6,loop)]
        %[kerV(1,indb(3)) kerV(2,indb(4)) ker2V(1,indb(5),:)' ker2V(2,indb(6),:)']


        
        %drawnow
%             end
%         end
        
   


