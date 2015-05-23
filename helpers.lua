-- helpers

function waitForEnter()
	local input
	repeat
		print("Press <Enter> to continue.")
		input = io.read()
	until string.lower(input) == ""
end

function getMnist()
	----------------------------------------------------------------------
	-- This script downloads and loads the MNIST dataset
	-- http://yann.lecun.com/exdb/mnist/
	----------------------------------------------------------------------

	-- Here we download dataset files.
	-- Note: files were converted from their original LUSH format
	-- to Torch's internal format.

	-- The SVHN dataset contains 3 files:
	--    + train: training data
	--    + test:  test data

	local tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
	local train_file = 'mnist.t7/train_32x32.t7'
	local test_file = 'mnist.t7/test_32x32.t7'
	local missing = false
	if not paths.filep(train_file) or not paths.filep(test_file) then
		missing = true
	end

	if missing and not paths.dirp('mnist.t7') then
		print '==> downloading MNIST dataset'
		os.execute('wget ' .. tar)
	end

	if missing then
		os.execute('tar xvf ' .. paths.basename(tar))
	end

	----------------------------------------------------------------------
	print '==> loading MNIST dataset'

	-- We load the dataset from disk, it's straightforward

	trainData = torch.load(train_file,'ascii')
	testData = torch.load(test_file,'ascii')

	print('Training Data:')
	print(trainData)
	print()

	print('Test Data:')
	print(testData)
	print()

	print '==> done'
	return trainData,testData
end