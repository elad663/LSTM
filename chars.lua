require 'io'
require 'model' 
require 'nngraph' 

params={model_file='rnn200.net',
		vocab_file='chars_vocab.t7b',
		gpu=1,
		batch_size=20,
		seq_length=20
}

g_init_gpu(params.gpu)
model=torch.load(params.model_file)
vocab=torch.load(params.vocab_file)
--print(vocab)

-- This function is based on given code
function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if #line > 1 then error({code="more than one character"}) end
  if vocab[line[1]] == nil then error({code="vocab", char = line}) end  
  return line
end


-- Adhering to batch size
function trans_input(char)
	local x = torch.zeros(1)
	x[1]= vocab[char[1]]
	x = x:resize(x:size(1), 1):expand(x:size(1), params.batch_size)
   return x
end



-- Getting probabilities
--[[
function next_probabilites(input)

  g_disable_dropout(model.rnns)
  perp = 0
  g_replace_table(model.s[0], model.start_s) --why?

  local predictions

  for i = 1, params.seq_length do
    x = input[pos]:cuda()	
	y=torch.ones(#x):cuda() --just rubbish
    s = model.s[i - 1]
    predictions,perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
	predictions=predictions[1]
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
	return(predictions)
end
--]]

function run_test()
  local perp = 0
  local predictions
  local x = input[1]:cuda()
  local y = torch.ones(#x):cuda()

  predictions,perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
  predictions=predictions[1]
  g_replace_table(model.s[0], model.s[1])
	return(predictions)
end

io.flush()
io.write("OK GO\n")
io.flush()
g_disable_dropout(model.rnns)
g_replace_table(model.s[0], model.start_s)


counter=1
while true do
	
	--- Sanity------- -------------------------------------------------------------
	ok, input = pcall(readline)
	if not ok then
		if input.code == "EOF" then
			--break -- end loop
		elseif input.code == "vocab" then
			io.flush()
			io.write("Character is not in vocabulary\n")
			io.flush()
		else
			io.flush()
			io.write("Should not get here\n")
			io.flush()
		end
	--- Sanity------- -------------------------------------------------------------
	else
		-- using dictionary and tranforming to batch size.
		input=trans_input(input)

		--x=next_probabilites(input)
		x=run_test(input)
		for i=1, x:size(1) do
			io.write(x[i] .. " ")
			io.flush()
		end		
		io.write('\n')
		io.flush()
	end

	if counter % 100000 == 0 then collectgarbage() end
	counter=counter+1
end








