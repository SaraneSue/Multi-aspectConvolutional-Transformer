clear
ReadPath = 'D:\class\UCAS-AIR\project\00experiment\rawdata\EOC-V\test\A10\';
SavePath = 'D:\class\UCAS-AIR\project\00experiment\data\EOC-V\test\A10\';
FileType = '*.020';
Files = dir([ReadPath FileType]);
NumberOfFiles = length(Files);
for i = 1 : NumberOfFiles
    FileName = Files(i).name;
    NameLength = length(FileName);
    FID = fopen([ReadPath FileName],'rb','ieee-be');
    ImgColumns = 0;
    ImgRows = 0;
    while ~feof(FID)                                % ��ȡPhoenixHeader������
        Text = fgetl(FID);
        if ~isempty(strfind(Text,'NumberOfColumns'))
            ImgColumns = str2double(Text(18:end));
            Text = fgetl(FID);
            ImgRows = str2double(Text(15:end));
            break;
        end
    end
    while ~feof(FID)                                 % ����PhoenixHeader
        Text = fgetl(FID);
        if ~isempty(strfind(Text,'[EndofPhoenixHeader]'))
            break
        end
    end
    Mag = fread(FID,ImgColumns*ImgRows,'float32','ieee-be');
    Img = reshape(Mag,[ImgColumns ImgRows]);
    Img = uint8(imadjust(Img)*255);
    %ͼƬ�ü����ü���ͼ��ߴ�Ϊ(w+1)x(h+1)��
    width = 67;
    height = 67;
    Img_tmp = imcrop(Img,[(ImgColumns-width)/2,(ImgRows-height)/2,width,height]);
    w = 63;
    h = 63;
    x = randperm(width-w,1);
    y = randperm(height-h,1);
    Img_crop = imcrop(Img_tmp,[y,x,w,h]);
    %����ͼƬ
    if exist(SavePath, 'dir')==0 %%�ж��ļ����Ƿ����
        mkdir(SavePath);  %%������ʱ�򣬴����ļ���
    end
    imwrite(Img_crop,[SavePath FileName(1:NameLength-3) 'jpg']);
    fclose(FID);
end
