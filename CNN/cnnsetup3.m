function net = cnnsetup3(net, x)
    inputmaps = 1;
    mapsize = size(squeeze(x(:, :, 1)));%������С,28*28 
    net.theta = [];
    for l = 1 : numel(net.layers)   %  layer,numel(net.layers)  ��ʾ�ж��ٲ�  
        if strcmp(net.layers{l}.type, 's')% ���������Ӳ����� 
            mapsize = mapsize / net.layers{l}.scale;
        end
        if strcmp(net.layers{l}.type, 'c') % �������Ǿ���� 
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
%             fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            for j = 1 : net.layers{l}.outputmaps  
%                 fan_in = inputmaps * net.layers{l}.kernelsize ^ 2; 
                for i = 1 : inputmaps 
%                     net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                    net.filter{l/2} = 1e-1*randn(net.layers{l}.kernelsize,...
                        net.layers{l}.kernelsize,net.layers{l}.outputmaps,inputmaps);
%                     net.theta = [net.theta ;net.filter{l/2}];
                end 
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    hidden = prod(mapsize) * inputmaps;
    onum = 10;%��������Ԫ����
    r  = sqrt(6)/sqrt(onum+hidden+1);
    net.w_net = rand(onum, hidden) * 2*r  - r;
%     f1 = net.filter{1, 1};
%     f2 = net.filter{1, 2};
    net.theta = [net.filter{1, 1}(:) ;net.filter{1, 2}(:); net.w_net(:)];
end
