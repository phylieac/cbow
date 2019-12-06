//
//  vocab.hpp
//  Pytorch
//
//  Created by 潘洪岩 on 2019/11/27.
//  Copyright © 2019 潘洪岩. All rights reserved.
//

#ifndef vocab_hpp
#define vocab_hpp

#include <stdio.h>
#include <iostream>
#include <string>
#include <map>


typedef std::pair<std::string, int> PAIR;

struct CmpByValue {
    bool operator()(const PAIR& lhs, const PAIR& rhs){
        return lhs.second > rhs.second;
    }
};
        
class Vocab
{
    private:
        std::map<std::string,int> vocab;
        long count=0;
    public:
        void pushw(std::string& w)
        {
            auto iter = vocab.find(w);
            if(iter==vocab.end())
                vocab.insert(std::make_pair(w, 1));
            else
                vocab[w]=iter->second+1;
            count++;
        }
        
        void sort()
        {
            std::vector<PAIR> svocab(vocab.begin(),vocab.end());
            std::sort(svocab.begin(),svocab.end(),CmpByValue());
            vocab.clear();
            int pos=0;
            for(PAIR p:svocab){
//                std::cout<<p.first<<"\t"<<p.second<<std::endl;
                vocab.insert(std::make_pair(p.first,pos));
                pos++;
            }
        }
        
        int get_id(std::string& w)
        {
            auto iter= vocab.find(w);
            return iter==vocab.end()?-1:iter->second;
        }
        
        std::map<std::string,int> export_vocab()
        {
            return vocab;
        }
        
        long tsize()
        {
            return count;
        }
    
        int size(){
            return vocab.size();
        }
};
        
#endif /* vocab_hpp */
