clear
ReadPath = 'D:\class\UCAS-AIR\02project\00experiment\rawdata\SOC\train\ZSU_23_4\';
SavePath = 'D:\class\UCAS-AIR\02project\00experiment\data\SOC-128\train\ZSU_23_4\';
FileType = '*.026';
Files = dir([ReadPath FileType]);
NumberOfFiles = length(Files);
for i = 1 : NumberOfFiles
    FileName = Files(i).name;
    NameLength = length(FileName);
    FID = fopen([ReadPath FileName],'rb','ieee-be');
    ImgColumns = 0;
    ImgRows = 0;
    while ~feof(FID)                                % 读取PhoenixHeader中数据
        Text = fgetl(FID);
        if ~isempty(strfind(Text,'NumberOfColumns'))
            ImgColumns = str2double(Text(18:end));
            Text = fgetl(FID);
            ImgRows = str2double(Text(15:end));
            break;
        end
    end
    while ~feof(FID)                                 % 跳过PhoenixHeader
        Text = fgetl(FID);
        if ~isempty(strfind(Text,'[EndofPhoenixHeader]'))
            break
        end
    end
    Mag = fread(FID,ImgColumns*ImgRows,'float32','ieee-be');
    Img = reshape(Mag,[ImgColumns ImgRows]);
    Img = uint8(imadjust(Img)*255);
    %图片裁剪（裁剪后图像尺寸为(w+1)x(h+1)）
    width = 127;
    height = 127;
    Img_tmp = imcrop(Img,[(ImgColumns-width)/2,(ImgRows-height)/2,width,height]);
%     w = 63;
%     h = 63;
%     x = randperm(width-w,1);
%     y = randperm(height-h,1);
%     Img_crop = imcrop(Img_tmp,[y,x,w,h]);
    %保存图片
    if exist(SavePath, 'dir')==0 %%判断文件夹是否存在
        mkdir(SavePath);  %%不存在时候，创建文件夹
    end
    imwrite(Img_tmp,[SavePath FileName(1:NameLength-3) 'jpg']);
    fclose(FID);
end
