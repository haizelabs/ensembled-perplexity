from vllm import LLM, SamplingParams
import os
import math
class PerplexityModel():

    def __init__(self, model_name, gpus=[0]):
        self.model_name = model_name
        self.gpus = gpus

        self.localize_visible_gpus()
        # Modify LLM params based on GPU type. Below params are optimized for L4 GPUs.
        self.llm = LLM(model=model_name, gpu_memory_utilization=0.7, enforce_eager=True, tensor_parallel_size=len(gpus), max_model_len=1024, max_num_seqs=16, swap_space=4)

    def localize_visible_gpus(self):
        # set cuda visble devices for VLLM ([0, 1, ...] -> '0,1')
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in self.gpus])

    def perplexity_from_logprobs(self, logprobs):        
        return math.exp(sum(logprobs) / (-len(logprobs)))

    def get_perplexity(self, prompts):
        """
        Convert a list of prompts to a list of perplexity scores for those prompts.
        Ex: ["Hello there", "This is a random sentence"] -> [8.424, 14.239]
        """
        self.localize_visible_gpus()
        sampling_params = SamplingParams(temperature=0, top_p=1, prompt_logprobs=1, max_tokens=1)
        outputs = self.llm.generate(prompts, sampling_params)

        perplexities = []
        for output in outputs:
            prompt_logprobs = []
            for prob in output.prompt_logprobs:
                if prob is not None:
                    log_prob = list(prob.values())[0].logprob
                    prompt_logprobs.append(log_prob)
            
            perplexity_output = self.perplexity_from_logprobs(prompt_logprobs)
            perplexities.append(perplexity_output)
            print(f'Perplexity: {perplexity_output} | Prompt: {output.prompt}')

        return perplexities
    
    