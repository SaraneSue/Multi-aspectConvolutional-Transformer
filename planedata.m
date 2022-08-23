clear
ReadPath = 'D:\class\UCAS-AIR\02project\00experiment\rawdata\plane\H1H1\';
SavePath = 'D:\class\UCAS-AIR\02project\00experiment\data\plane\AT504-2\';
FileType = '*.mat';
Files = dir([ReadPath FileType]);
NumberOfFiles = length(Files);
for i = 1 : NumberOfFiles
    FileName = Files(i).name;
    NameLength = length(FileName);
    FID = load([ReadPath FileName]);
    fig = FID.fig.Image;
    %ȡģ
    Img = abs(fig);
    %�ü�
    %PLANE = imcrop(Img,[450 850 512 512]);
    %PLANE = imcrop(Img,[460 1335 512 512]);
    PLANE = imcrop(Img,[495 1815 512 512]);
    %PLANE = imcrop(Img,[490 2300 512 512]);
    %��ֵ
    maxnum = max(max(PLANE));
    top = 0.05*maxnum;
    PLANE(PLANE>top) = top;
    %��ֵ
    PLA = imresize(PLANE,[64 64],'bilinear');
    %��һ��
    ymax=255;ymin=0;
    xmax = max(max(PLA));
    xmin = min(min(PLA));
    plane = round((ymax-ymin)*(PLA-xmin)/(xmax-xmin) + ymin);
    %s = imshow(plane,[]);
    %����ͼƬ
    if exist(SavePath, 'dir')==0 %%�ж��ļ����Ƿ����
        mkdir(SavePath);  %%������ʱ�򣬴����ļ���
    end
    imwrite(mat2gray(plane),[SavePath FileName(1:NameLength-3) 'jpg']);
end
