%% 
function net = cnnsetup2(net, x)
    inputmaps = 1;
    mapsize = size(squeeze(x(:, :, 1))); %输入层大小,28*28 
    for l = 1 : numel(net.layers)   % 表示有多少层  
        if strcmp(net.layers{l}.type, 's')% 如果这层是子采样层 
            mapsize = mapsize / net.layers{l}.scale;
        end
        if strcmp(net.layers{l}.type, 'c') % 如果这层是卷积层 
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            for j = 1 : net.layers{l}.outputmaps  
                for i = 1 : inputmaps 

                    net.filter{l/2} = 1e-1*randn(net.layers{l}.kernelsize,...
                        net.layers{l}.kernelsize,net.layers{l}.outputmaps);
                end 
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    hidden = prod(mapsize) * inputmaps;
    onum = 10;
    r  = sqrt(6)/sqrt(onum+hidden+1);
    net.w_net = rand(onum, hidden) * 2*r  - r;
    net.theta = [net.filter{1}(:) ; net.w_net(:)];
end
