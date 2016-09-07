--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

stringx = require('pl.stringx')
file = require('pl.file')

ptb_path = "./data/"


--[[
local trainfn = ptb_path .. "ptb.train.txt"
local testfn  = ptb_path .. "ptb.test.txt"
local validfn = ptb_path .. "ptb.valid.txt"
--]]


trainfn = ptb_path .. "ptb.char.train.txt"
validfn = ptb_path .. "ptb.char.valid.txt"


vocab_idx = 0
vocab_map = {}

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
function replicate(x_inp, batch_size)
   local s = x_inp:size(1)
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   for i = 1, batch_size do
     local start = torch.round((i - 1) * s / batch_size) + 1
     local finish = start + x:size(1) - 1
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
   end
   return x
end

function load_data(fname)
   local data = file.read(fname)
   data = stringx.replace(data, '\n', '<eos>')
   data = stringx.split(data)
   --print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.zeros(#data)
   for i = 1, #data do
      if vocab_map[data[i]] == nil then
         vocab_idx = vocab_idx + 1
         vocab_map[data[i]] = vocab_idx
      end
      x[i] = vocab_map[data[i]]
   end
   return x
end

-- I wrote this function to convert the characters b/c I
-- didn't see you already prepared the files. this fun is not used. 
function load_data_new(fname)
   local data = file.read(fname)
   data = stringx.replace(data, '\n', '.')
   --print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.zeros(#data)
	for i=1, #data do
		char=data:sub(i,i)
		  if vocab_map[char] == nil then
		     vocab_idx = vocab_idx + 1
		     vocab_map[char] = vocab_idx
		  end
		  x[i] = vocab_map[char]
	end
   return x
end

function traindataset(batch_size, char)
   local x = load_data(trainfn)
   x = replicate(x, batch_size)
   return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
function testdataset(batch_size)
   if testfn then
      local x = load_data(testfn)
      x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
      return x
   end
end

function validdataset(batch_size)
   local x = load_data(validfn)
   x = replicate(x, batch_size)
   return x
end

return {traindataset=traindataset,
        testdataset=testdataset,
        validdataset=validdataset,
        vocab_map=vocab_map}
