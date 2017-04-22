load train_images/digitStruct.mat

for i = 1:length(digitStruct)
    im = imread(['train_images/' digitStruct(i).name]);
    
    for j = 1:length(digitStruct(i).bbox)
        [height, width, z] = size(im);

        bbox = digitStruct(i).bbox(j);
        
        top = max(bbox.top + 1, 1);
        bottom = min(bbox.top + bbox.height, height);
        
        left = max(bbox.left + 1, 1);
        right = min(bbox.left + bbox.width, width);
        
        imwrite(im(top:bottom, left:right, :), ['training_set/' num2str(bbox.label) '/' num2str(i) num2str(j) '.png']);
    end
end