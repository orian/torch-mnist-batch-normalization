require('torch')
require('nn')
require('gnuplot')
require('os')
require('helpers')
require('optim')

dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Batch Normalization example')
cmd:text()
cmd:text('Options:')
cmd:option('-printEverySec', 30, 'print interval')
cmd:option('-batch_size', 32, 'how big batch to use')
cmd:option('-max_epoch', 100, 'number of epochs')
cmd:option('-learningRate', 0.1, 'learning rate')
cmd:option('-weightDecay', 0.0, 'weight decay')
cmd:option('-momentum', 0.0, 'number of epochs')
cmd:option('-cuda', false, 'use CUDA')
cmd:option('-retrain', '', 'a model to load')
cmd:option('-onlyconv', false, 'only convolution')
cmd:option('-onlyfull', false, 'only full models')
opt = cmd:parse(arg)

print(opt)

if opt.cuda then
  require('cunn')
  require('cutorch')
  defTensorType = 'torch.CudaTensor'
  print('using CUDA')
else
  defTensorType = 'torch.FloatTensor'
  print('using CPU')
end

torch.setdefaulttensortype(defTensorType)

print('num threads: ', torch.getnumthreads())

trainData,testData = getMnist()

dataSize = trainData.data:size(1)

-- train
trainData.data = trainData.data:type(defTensorType)
trainData.labels = trainData.labels:type(defTensorType)
-- test
testData.data = testData.data:type(defTensorType)
testData.labels = testData.labels:type(defTensorType)

classes = {'1','2','3','4','5','6','7','8','9','0'}

function CreateConvModel(mode)
    local model=nn.Sequential();
    -- nInputPlane, nOutputPlane, kW, kH, [dW], [dH],  [padding]
    -- if mode == 1 then
    --   model:add(nn.SpatialBatchNormalization(3))
    -- end
    model:add(nn.SpatialConvolutionMM(1, 32, 4, 4)) -- output size: 29x29
    if mode == 2 then
      model:add(nn.SpatialBatchNormalization(32))
    end
    model:add(nn.ReLU(true))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- output size 14x14

    -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
    if mode == 1 then
      model:add(nn.SpatialBatchNormalization(32))
    end
    model:add(nn.SpatialConvolutionMM(32, 64, 5, 5)) -- output size: 10x10
    if mode == 2 then
      model:add(nn.SpatialBatchNormalization(64))
    end
    model:add(nn.ReLU(true))
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- 5x5

    if mode == 1 then
      model:add(nn.SpatialBatchNormalization(64))
    end
    -- stage 3 : standard 2-layer model:
    model:add(nn.View(64*5*5))
    model:add(nn.Linear(64*5*5, 200))
    if mode == 2 then
      model:add(nn.BatchNormalization(200))
    end
    model:add(nn.ReLU(true))

    if mode == 1 then
      model:add(nn.BatchNormalization(200))
    end
    model:add(nn.Linear(200, #classes))
    if mode == 2 then
      model:add(nn.BatchNormalization(#classes))
    end
    local criterion = nn.CrossEntropyCriterion()
    return model, criterion
end

-- mode 0 linear, 1 BatchNormalization for layer input, 2 BN for nonlinearity
-- function input.
function CreateSimpleModel(mode)
    torch.setdefaulttensortype('torch.FloatTensor')
    after = after or false
    ----------------------------------------------------------------------
    -- define model to train
    -- on the 10-class classification problem
    --
    local model=nn.Sequential();  -- make a multi-layer perceptron

    -- nInputPlane, nOutputPlane, kW, kH, [dW], [dH],  [padding]
    local nHidden = 100
    local inSize = 32*32
    model:add(nn.View(-1,inSize))
    if mode == 1 then
      model:add(nn.BatchNormalization(inSize))
    end
    if mode == 3 then
      model:add(nn.Add(inSize))
      model:add(nn.CMul(inSize))
    end
    model:add(nn.Linear(inSize, nHidden))
    if mode == 2 then
      model:add(nn.BatchNormalization(nHidden))
    end
    model:add(nn.Sigmoid())

    local n = 2
    for i=1,n do
      if mode == 1 then
        model:add(nn.BatchNormalization(nHidden))
      end
      if mode == 3 then
        model:add(nn.Add(nHidden))
        model:add(nn.CMul(nHidden))
      end
      model:add(nn.Linear(nHidden, nHidden))
      if mode == 2 then
        model:add(nn.BatchNormalization(nHidden))
      end
      model:add(nn.Sigmoid())
    end

    if mode == 1 then
      model:add(nn.BatchNormalization(nHidden))
    end
    if mode == 3 then
      model:add(nn.Add(nHidden))
      model:add(nn.CMul(nHidden))
    end
    model:add(nn.Linear(nHidden, #classes))
    if mode == 2 then
      model:add(nn.BatchNormalization(#classes))
    end
    model:add(nn.Sigmoid())

    local criterion = nn.CrossEntropyCriterion()
    if opt.cuda then
      torch.setdefaulttensortype('torch.CudaTensor')
      return model:cuda(), criterion:cuda()
    end
    return model, criterion
end

local batchSize = opt.batch_size
local maxEpoch = opt.max_epoch
local printConfusionEvery = 20
local batchNum = dataSize/batchSize
print('number of batches: ', batchNum)
local printEverySec = opt.printEverySec or 30  -- how often to print

function train(model, criterion, learningRate)
  collectgarbage()
  local parameters,gradParameters = model:getParameters()

  local optimConfig = {learningRate = learningRate or opt.learningRate,
                  weightDecay = opt.weightDecay,
                  momentum = opt.momentum,
                  learningRateDecay = 5e-7}
  local optimState = {}
  local confusion = optim.ConfusionMatrix(classes)
  local costs = {}

  for epoch = 1,maxEpoch do
    local printErr = 0  -- accumulate error between print
    local printBatchIdx = 1 -- need to know how many inputs between printing
    local lastPrint = sys.clock()

    local cost = 0
    local batchIdx = 0
    for batchStart = 1,dataSize-batchSize,batchSize do
      batchIdx = batchIdx + 1
      local batchData = trainData.data:narrow(1, batchStart, batchSize)
      local batchLabels = trainData.labels:narrow(1, batchStart, batchSize)
      local feval = function(x)
          -- get new parameters
        if x ~= parameters then
          parameters:copy(x)
        end
        -- reset gradients
        gradParameters:zero()

        -- estimate f
        local output = model:forward(batchData)
        -- f is the average of all criterions
        local f = criterion:forward(output, batchLabels)
        cost = cost + f
        printErr = printErr + f

        confusion:batchAdd(model.output, batchLabels)

        -- estimate df/dW
        local df_do = criterion:backward(output, batchLabels)
        model:backward(batchData, df_do)

        if sys.clock()-lastPrint >= printEverySec then
          local ns = batchStart-printBatchIdx+batchSize
          print(string.format("<trainer> miniBatch = %d(%d inputs) error = %.6f", batchIdx, batchIdx*batchSize, printErr/ns))
          print(confusion)
          confusion:zero()
          printErr = 0
          printBatchIdx = batchStart+batchSize
          lastPrint = sys.clock()
        end
        gradParameters:div(batchSize)
        f = f/batchSize
        return f,gradParameters
      end
      optim.sgd(feval, parameters, optimConfig, optimState)
    end

    print(string.format('Epoch: %.4d/%d, cost: %0.6f', epoch, maxEpoch, cost/(batchIdx*batchSize)))
    if printBatchIdx == 1 then
      print(confusion)
    end
    confusion:zero()
    table.insert(costs, cost/(batchIdx*batchSize))
  end
  return costs
end

local learningRates = {0.5, 1.3, 1.3, 0.5, 0.5,1.3,1.3}
local costs = {}
if not opt.onlyconv then
  for i = 3,3 do
    local model,criterion = CreateSimpleModel(i)
    table.insert(costs, train(model, criterion, learningRates[i+1]))
  end
end

if not opt.onlyfull then
  for i=0,2 do
    local model, criterion = CreateConvModel(i)
    table.insert(costs, train(model, criterion, learningRates[i+4]))
  end
end

costs = torch.FloatTensor(costs)
torch.setdefaulttensortype('torch.FloatTensor')
fn=os.date('%d_%m_%y %H_%M.png')
print('saving plot into: '..fn)
gnuplot.pngfigure(fn)
if opt.onlyconv then
  gnuplot.plot({'Conv',costs[1]},{'Conv before',costs[2]}, {'Conv after',costs[3]})
  gnuplot.plotflush()
  gnuplot.title('Errors in epochs')
elseif opt.onlyfull then
  gnuplot.plot({'Full',costs[1]},{'BN input',costs[2]}, {'BN before sigmoid',costs[3]},
               {'MulAdd',costs[4]})
  gnuplot.plotflush()
  gnuplot.title('Errors in epochs')
else
  gnuplot.plot({'Full',costs[1]},{'BN input',costs[2]}, {'BN before sigmoid',costs[3]},
               {'ConvNet', costs[4]}, {'Conv before',costs[5]}, {'Conv after',costs[6]})
  gnuplot.plotflush()
  gnuplot.title('Errors in epochs')
end
