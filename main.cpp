//
//  main.cpp
//  Pytorch
//
//  Created by 潘洪岩 on 2019/9/3.
//  Copyright © 2019 潘洪岩. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <functional>
#include "data/data.h"
#include "embedding/embedding.h"

std::string TRAINFILE("/Users/panhongyan/bert/sample_text.txt");

template <typename DataLoader>
void train(WordEmbedding& emb,torch::data::datasets::Options& options,DataLoader& loader,torch::optim::Optimizer& optimizer,size_t epoch,size_t data_size)
{
    size_t index = 0;
    emb.train();
    float Loss = 0;
    for (auto& batch : loader) {
        auto data = batch.data.to(options.device);
        auto targets = batch.target.to(options.device).view({-1});
        auto output = emb.forward(data);
        auto loss = torch::nll_loss(output, targets);
        assert(!std::isnan(loss.template item<float>()));
        emb.zero_grad();
        loss.backward();
        optimizer.step();
        
        Loss += loss.template item<float>();
//        std::cout<<index<<std::endl;
        if (index++ % options.log_interval == 0) {
            auto end = std::min(data_size, (index + 1) * options.train_batch_size);
            std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
            << "\tLoss: " << Loss / end << std::endl;
        }
    }
}

void test(WordEmbedding& wemb,Vocab& vocab,std::string& word)
{
    long id=vocab.get_id(word);
    
}

int main() {
    torch::manual_seed(1);
    torch::data::datasets::Options opts;
    auto corpus= torch::data::datasets::loadTrain(TRAINFILE);
    torch::data::datasets::EmbeddingTextData text(corpus.data,corpus.vocab,opts);
    auto train_text=text.map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_text), torch::data::DataLoaderOptions().batch_size(opts.train_batch_size).workers(2));
    auto wemb =std::make_shared<WordEmbedding>(corpus.vocab.size(),opts.dim);
    wemb->to(opts.device);
    torch::optim::Adam adam(wemb->parameters(),torch::optim::AdamOptions(0.01));
    size_t data_size=corpus.vocab.tsize();
    for (size_t epoch=0;epoch<opts.epoch;epoch++) {
        train(*wemb, opts, *train_loader, adam, epoch, data_size);
        std::cout << std::endl;
    }
    
    torch::save(wemb, "/Users/panhongyan/cbow/cke.pt");
    return 0;
}
