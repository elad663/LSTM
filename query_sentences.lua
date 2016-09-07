require 'io'
require 'model' 
require 'nngraph' 
require 'base' 

params={model_file='words_model.net',
		vocab_file='words_vocab.t7b',
		gpu=4,
		batch_size=20
}

g_init_gpu(params.gpu)
model=torch.load(params.model_file)
vocab=torch.load(params.vocab_file)


-- This function is based on given code
function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if vocab[line[i]] == nil then error({code="vocab", word = line[i]}) end
  end
  return line
end


-- Adhering to batch size
function trans_input(lines)
	local x = torch.zeros((#lines-1))
	for i=1, x:size(1) do
		x[i]= vocab[lines[i+1]]
	end
	x = x:resize(x:size(1), 1):expand(x:size(1), params.batch_size)
   return x
end


-- Getting probabilities
function next_probabilites(input)

  g_disable_dropout(model.rnns)
  perp = 0
  len = input:size(1)
  g_replace_table(model.s[0], model.start_s) --why?

  local predictions

  for i = 1, (len) do
    x = input[i]:cuda()	
	y=torch.ones(#x):cuda() --just rubbish
    s = model.s[i - 1]
    predictions,perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
	predictions=predictions[1]
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
	return(predictions)
end



function query_sentences(howmany)

	for p=1, howmany do
	  print("Query: len word1 word2 etc")
	  ok, line = pcall(readline)
	 if not ok then
		if line.code == "EOF" then
		  break -- end loop
		elseif line.code == "vocab" then
		  print("Word not in vocabulary: ", line.word)
		elseif line.code == "init" then
		  print("Start with a number")
		else
		  print("should not be here")
		end
	  else
		sen_len=(#line)-1
		--print("OK " .. line[1] .. " words to Predict with " .. sen_len .. " words")
		--print(line)
		for j=1, line[1] do
			-- make it into an integer and coherce to batches
			input=trans_input(line)
		
			--making a prediction
			x=torch.exp(next_probabilites(input))
			i=torch.multinomial(x:float(),1,false)

			-- when it was deterministic
			--_,i=x:max(1)

			--finding the table key...
			counter=1
			for k,v in pairs(vocab) do
				if counter==i[1] then 
					prediction=k
				end
				counter=counter+1 
			end
			--print(prediction)		
			-- add the prediction to the input table
			nextkey=sen_len+j+1
			line[nextkey]=prediction
			io.write(prediction .. " ")
		end
		io.write('\n')
	  end
	end
end









