//INITIALISATION OF ALL REQUIRED OBJECTS :

//create objects to extract the data from the canvas later :
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');

//define an object to change the content of the text :
const displayElement = document.getElementById('displayText');

// Initialize canvas
ctx.fillStyle = '#000'; // Set initial color to white
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Handle mouse drawing on the canvas
let isDrawing = false;

//load the NN as async function to be sure not to activate anything else before the end of the loading
async function loadModel() {
    model = new onnx.InferenceSession();
    await model.loadModel('emnist.onnx');
  }  
loadModel()



//DEFINITION OF ALL THE USER INTERACTIONS :

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath(); 
}

function draw(e) {
    if (!isDrawing) return;

    ctx.lineWidth = 10; 
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#FFF'; 


    //get user position no matter if it's a mouse on laptop or finger on mobile device :
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    //draw the line to target coords :
    ctx.lineTo(clientX - canvas.offsetLeft, clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(clientX - canvas.offsetLeft, clientY - canvas.offsetTop);
}

// Event listeners for drawing on laptop
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseout', stopDrawing);

// Event listeners for drawing on mobile
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchmove', draw);

//Block the default usage of moving user's finger on screen for our draw function (to prevent him from scrolling at the same time for exemple)
canvas.addEventListener('touchstart', function (e) {e.preventDefault();});
canvas.addEventListener('touchmove', function (e) {e.preventDefault();});


//Fonction linked to "Reset" button :
function resetCanvas() {
    ctx.fillStyle = '#000'; 
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    displayElement.textContent = "execute to recognition to start";
}

//Fonction linked to "Recognize" button :
function Recognize() {
    let toprint = predict()
    console.log(toprint)
}

//This fonction reads the content of the canvas, makes all the necessary transformation and applies the NN model to it :
async function predict() {
   
    //This creates an other canvas which rescales the on-screen one
    let tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 28;
    tmpCanvas.height = 28;
    let tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(canvas, 0, 0, 28, 28);
  
    //the next block creates a gray hollow around the line drown by the user as in the dataset used to train the model it was in gray scale and not in black or white
    let NOT_ZERO_NUMB = 0
    let pix_num = 0
    let imgData = tmpCtx.getImageData(0, 0, 28, 28).data;
    for (let i = 0; i < imgData.length; i+=1){
        if (imgData[i]!=0){
            NOT_ZERO_NUMB += 1; }
        pix_num += 1
    }
    let input = new Float32Array(28 * 28);
    for (let i = 0; i < imgData.length; i += 4) {
        let grayscale = 0
        if (imgData[i]!=0)  {grayscale = 255}
        if (imgData[i+1]!=0)  {grayscale = 255}
        if (imgData[i+2]!=0)  {grayscale = 255}
        grayscale = (((grayscale/ 255)) - 0.1736) / 0.3317;
        input[i/4] = grayscale
    }
    //we round all float inferior to 1 to 0 to avoid problems during calculation
    for (let i = 0; i < input.length; i += 1) {
      if (input[i] < 1) {
        input[i] = 0;
      }
    }
  
    //The EMNIST dataset is transposed compare to what we see on screen, so we have to transpose our input too
    let transposedInput = new Float32Array(28*28)
    for (let y = 0; y < 28; y++){
        for (let x = 0; x < 28; x++){
            transposedInput[x * 28 + y] = input[y * 28 +x]
        }    
    }

    //Execution of the model :
    let TensorInput = new onnx.Tensor(transposedInput, 'float32',[1,1,28,28]);
    let outputMap = await model.run([TensorInput]);
    let outputData = outputMap.values().next().value.data;
    let predict_char = String.fromCharCode(outputData.indexOf(Math.max(...outputData))+64);
    

    //return the result
    displayElement.textContent = "I think that you draw a(n) : " + predict_char;
    return outputData.indexOf(Math.max(...outputData));
  }