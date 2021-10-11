function use_imgFun(pred_dataL, pred_dataR){
  // https://www.w3schools.com/tags/canvas_createimagedata.asp
  // console.log((pred_data[0][0][0]));
  width = pred_dataL[0].length;
  height = pred_dataL.length;
var cRight = document.getElementById("rightEye");
var cLeft = document.getElementById("leftEye");
var ctxLeft = cLeft.getContext("2d");
var ctxRight = cRight.getContext("2d");
var imgDataLeft = ctxLeft.createImageData(width, height);
var imgDataRight = ctxRight.createImageData(width, height);

// console.log(width);
// console.log(height);
// console.log(imgData.data.length);
var i;
var j;
var k = 0;
for (i = 0; i < imgDataLeft.data.length; ) {
  // console.log('row ====>',(i/4)%height);
  // console.log(i%width);
       // console.log("check the imageData ==>",i);
       // console.log('row ===>',k);
       for (var j = 0; j < width; j++) {
         // console.log('column==>',j);
       imgDataLeft.data[i+j*4+0] = pred_dataL[k][j][0];
       imgDataLeft.data[i+j*4+1] = pred_dataL[k][j][1];
       imgDataLeft.data[i+j*4+2] = pred_dataL[k][j][2];
       imgDataLeft.data[i+j*4+3] = 255;

       imgDataRight.data[i+j*4+0] = pred_dataR[k][j][0];
       imgDataRight.data[i+j*4+1] = pred_dataR[k][j][1];
       imgDataRight.data[i+j*4+2] = pred_dataR[k][j][2];
       imgDataRight.data[i+j*4+3] = 255;
}
        i = i+width*4

k = k+1;
// console.log(i,j);
}

ctxLeft.putImageData(imgDataLeft, 10, 10);
ctxRight.putImageData(imgDataRight, 10, 10);
}
