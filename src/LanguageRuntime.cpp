#include "headers/LanguageRuntime.h"
#include "headers/TextTokenizer.h"
#include "headers/TransformerLM.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <cstdio>
#include <optional>

namespace {
std::string readText(const std::string& path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Unable to open text file: "+path);std::string s((std::istreambuf_iterator<char>(f)),{});return TextTokenizer::normalizeText(s);}
std::vector<std::string> lines(const std::string& text){std::vector<std::string> out;std::size_t start=0;while(start<text.size()){auto end=text.find('\n',start);if(end==std::string::npos)end=text.size();auto line=text.substr(start,end-start);if(!line.empty())out.push_back(line);start=end+1;}return out;}
std::vector<std::pair<std::string,std::string>> qaPairs(const std::string&text){std::vector<std::pair<std::string,std::string>> result;for(const auto&line:lines(text)){const auto split=line.find(" | ");if(split==std::string::npos)throw std::runtime_error("QA rows must use: prompt | reference answer");auto prompt=line.substr(0,split),answer=line.substr(split+3);if(prompt.empty()||answer.empty())throw std::runtime_error("QA prompt and answer must be non-empty");result.emplace_back(std::move(prompt),std::move(answer));}return result;}
std::vector<LanguageSequence> windowSequences(const std::vector<std::uint32_t>&ids,std::size_t context,std::size_t loss_start){
    std::vector<LanguageSequence> out;const std::size_t stride=std::max<std::size_t>(1,context/2);
    for(std::size_t target_start=0;target_start+1<ids.size();target_start+=stride){
        const std::size_t target_end=std::min(target_start+stride,ids.size()-1);
        const std::size_t input_start=target_start>context-stride?target_start-(context-stride):0;
        LanguageSequence sequence;sequence.input.assign(ids.begin()+input_start,ids.begin()+target_end);sequence.target.assign(ids.begin()+input_start+1,ids.begin()+target_end+1);
        sequence.loss_mask.reserve(sequence.target.size());bool supervised=false;
        for(std::size_t i=0;i<sequence.target.size();++i){const auto input_position=input_start+i;const bool include=input_position>=target_start&&input_position+1>=loss_start;sequence.loss_mask.push_back(include?1:0);supervised=supervised||include;}
        if(supervised)out.push_back(std::move(sequence));
    }
    return out;
}
std::vector<LanguageSequence> sequences(const TextTokenizer&t,const std::string& text,std::size_t context){std::vector<LanguageSequence> out;for(auto&line:lines(text)){auto ids=t.encode(line);ids.insert(ids.begin(),TextTokenizer::bosId);ids.push_back(TextTokenizer::eosId);auto windows=windowSequences(ids,context,0);out.insert(out.end(),std::make_move_iterator(windows.begin()),std::make_move_iterator(windows.end()));}return out;}
std::vector<LanguageSequence> qaSequences(const TextTokenizer&t,const std::vector<std::pair<std::string,std::string>>&pairs,std::size_t context){std::vector<LanguageSequence> out;for(const auto&pair:pairs){auto prompt=t.encode(pair.first),answer=t.encode(pair.second);if(answer.empty())throw std::runtime_error("QA answer produces no tokens");std::vector<std::uint32_t>ids{TextTokenizer::bosId};ids.insert(ids.end(),prompt.begin(),prompt.end());const std::size_t prompt_length=ids.size();ids.insert(ids.end(),answer.begin(),answer.end());ids.push_back(TextTokenizer::eosId);auto windows=windowSequences(ids,context,prompt_length);out.insert(out.end(),std::make_move_iterator(windows.begin()),std::make_move_iterator(windows.end()));}return out;}
std::string value(int& i,int argc,char**argv,const std::string& option){if(i+1>=argc)throw std::invalid_argument("Missing value for "+option);return argv[++i];}
double number(const std::string&s,const std::string&n){std::size_t used=0;double x=std::stod(s,&used);if(used!=s.size()||!std::isfinite(x))throw std::invalid_argument("Invalid value for "+n);return x;}
void save(const std::string&path,const TextTokenizer&t,const TransformerLM&m){const std::string tmp=path+".tmp";{std::ofstream f(tmp,std::ios::binary|std::ios::trunc);if(!f)throw std::runtime_error("Unable to write model: "+tmp);const char magic[8]={'G','L','O','O','M','L','M','2'};f.write(magic,8);t.write(f);m.writeCheckpoint(f);f.flush();if(!f)throw std::runtime_error("Unable to flush model checkpoint");}if(std::rename(tmp.c_str(),path.c_str())!=0)throw std::runtime_error("Unable to replace model file: "+path);}
struct Model{TextTokenizer tokenizer;TransformerLM model;};
Model load(const std::string&path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Unable to open language model: "+path+" (train it first with 'make language-model-example' or 'gloomy lm train')");char magic[8],expected[8]={'G','L','O','O','M','L','M','2'};f.read(magic,8);if(!f||!std::equal(magic,magic+8,expected))throw std::runtime_error("Unsupported GLOOMY language model format");auto tok=TextTokenizer::read(f);auto model=TransformerLM::readCheckpoint(f);if(tok.vocabularySize()!=model.vocabularySize())throw std::runtime_error("Tokenizer and Transformer vocabulary sizes differ");return {std::move(tok),std::move(model)};}
std::vector<std::uint32_t> sample(const std::vector<double>&logits,double temperature,int top_k,double top_p,std::mt19937&rng){std::vector<std::size_t> ids(logits.size());std::iota(ids.begin(),ids.end(),0);if(temperature<=0)return {static_cast<std::uint32_t>(std::distance(logits.begin(),std::max_element(logits.begin(),logits.end())))};std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return logits[a]>logits[b];});if(top_k>0&&ids.size()>static_cast<std::size_t>(top_k))ids.resize(top_k);double mx=logits[ids.front()]/temperature;std::vector<double>w;double sum=0;for(auto id:ids){double x=std::exp(logits[id]/temperature-mx);w.push_back(x);sum+=x;}if(top_p<1&&top_p>0){double c=0;std::size_t keep=0;do{c+=w[keep]/sum;++keep;}while(keep<w.size()&&c<top_p);ids.resize(keep);w.resize(keep);sum=std::accumulate(w.begin(),w.end(),0.0);}for(auto&x:w)x/=sum;std::discrete_distribution<std::size_t>d(w.begin(),w.end());return {static_cast<std::uint32_t>(ids[d(rng)])};}
std::string canonicalAnswer(std::string text){text=TextTokenizer::normalizeText(text);std::string out;bool space=true;for(unsigned char c:text){if(c==' '||c=='\t'||c=='\r'||c=='\n'){if(!space)out.push_back(' ');space=true;}else{out.push_back(static_cast<char>(c>='A'&&c<='Z'?c+('a'-'A'):c));space=false;}}if(!out.empty()&&out.back()==' ')out.pop_back();return out;}
std::vector<std::string> canonicalTokens(const std::string&answer){std::string normalized=canonicalAnswer(answer);for(char&value:normalized)if(std::ispunct(static_cast<unsigned char>(value)))value=' ';std::istringstream input(normalized);std::vector<std::string> result;for(std::string token;input>>token;)result.push_back(token);return result;}
double tokenF1(const std::string&generated,const std::string&reference){const auto predicted=canonicalTokens(generated),expected=canonicalTokens(reference);if(predicted.empty()||expected.empty())return predicted.empty()&&expected.empty()?1.0:0.0;std::unordered_map<std::string,std::size_t> counts;for(const auto&token:expected)++counts[token];std::size_t overlap=0;for(const auto&token:predicted){auto found=counts.find(token);if(found!=counts.end()&&found->second){++overlap;--found->second;}}if(!overlap)return 0.0;const double precision=static_cast<double>(overlap)/predicted.size(),recall=static_cast<double>(overlap)/expected.size();return 2.0*precision*recall/(precision+recall);}
struct Completion{std::string text;std::size_t token_count;};
Completion greedyCompletion(const TextTokenizer&tok,const TransformerLM&model,const std::string&prompt,std::size_t max_tokens){auto ids=tok.encode(prompt);ids.insert(ids.begin(),TextTokenizer::bosId);std::vector<std::uint32_t> generated;for(std::size_t i=0;i<max_tokens;++i){auto logits=model.nextTokenLogits(ids);auto next=static_cast<std::uint32_t>(std::distance(logits.begin(),std::max_element(logits.begin(),logits.end())));if(next==TextTokenizer::eosId)break;ids.push_back(next);generated.push_back(next);}const std::size_t count=generated.size();return {tok.decode(generated),count};}
double continuationLogProbability(const TextTokenizer&tok,const TransformerLM&model,const std::string&prompt,const std::string&continuation){auto ids=tok.encode(prompt);ids.insert(ids.begin(),TextTokenizer::bosId);auto targets=tok.encode(continuation);if(targets.empty())throw std::runtime_error("Closed QA label produces no tokens");targets.push_back(TextTokenizer::eosId);double score=0;for(auto target:targets){const auto logits=model.nextTokenLogits(ids);if(target>=logits.size())throw std::runtime_error("Closed QA label token is outside the model vocabulary");const double maximum=*std::max_element(logits.begin(),logits.end());double sum=0;for(double value:logits)sum+=std::exp(value-maximum);score+=logits[target]-maximum-std::log(sum);ids.push_back(target);}return score;}
void usage(){std::cerr<<"Usage:\n  gloomy lm train --train FILE --validation FILE --model FILE [--qa-train FILE --vocab N --context N --dimension N --heads N --layers N --batch-size N --epochs N --learning-rate X --seed N]\n  gloomy lm generate --model FILE --prompt TEXT [--tokens N --temperature X --top-k N --top-p X --seed N]\n  gloomy lm evaluate --model FILE --text FILE [--qa FILE --qa-label LABEL ... --weight-precision int8|int16]\n";}
}

int runLanguageCommand(int argc,char*argv[]){try{if(argc<3){usage();return 2;}const std::string command=argv[2];if(command=="train"){
    std::string train_path,val_path,qa_train_path,model_path;std::size_t vocab=512,batch=1,epochs=10;TransformerConfig c;bool resume=false,has_model_option=false;
    for(int i=3;i<argc;++i){std::string a=argv[i];if(a=="--train")train_path=value(i,argc,argv,a);else if(a=="--validation")val_path=value(i,argc,argv,a);else if(a=="--qa-train")qa_train_path=value(i,argc,argv,a);else if(a=="--model")model_path=value(i,argc,argv,a);else if(a=="--resume")resume=true;else if(a=="--vocab"){has_model_option=true;vocab=std::stoul(value(i,argc,argv,a));}else if(a=="--context"){has_model_option=true;c.context_length=std::stoul(value(i,argc,argv,a));}else if(a=="--dimension"){has_model_option=true;c.model_dimension=std::stoul(value(i,argc,argv,a));}else if(a=="--heads"){has_model_option=true;c.attention_heads=std::stoul(value(i,argc,argv,a));}else if(a=="--layers"){has_model_option=true;c.layer_count=std::stoul(value(i,argc,argv,a));}else if(a=="--batch-size")batch=std::stoul(value(i,argc,argv,a));else if(a=="--epochs")epochs=std::stoul(value(i,argc,argv,a));else if(a=="--learning-rate"){has_model_option=true;c.learning_rate=number(value(i,argc,argv,a),a);}else if(a=="--seed"){has_model_option=true;c.seed=std::stoul(value(i,argc,argv,a));}else throw std::invalid_argument("Unknown option: "+a);}
    if(train_path.empty()||val_path.empty()||model_path.empty()||!batch||!epochs)throw std::invalid_argument("train requires --train, --validation, --model, positive batch size and epochs");
    auto tr=readText(train_path),va=readText(val_path);const auto train_lines=lines(tr),validation_lines=lines(va);const auto qa_train=qa_train_path.empty()?std::vector<std::pair<std::string,std::string>>{}:qaPairs(readText(qa_train_path));std::unordered_set<std::string> train_documents(train_lines.begin(),train_lines.end());for(const auto&qa:qa_train){train_documents.insert(qa.first);train_documents.insert(qa.second);}for(const auto&document:validation_lines)if(train_documents.count(document))throw std::invalid_argument("Exact text overlap between training and validation corpora");std::optional<TextTokenizer> tok;std::optional<TransformerLM> model;
    if(resume){if(has_model_option)throw std::invalid_argument("Model and tokenizer options are stored in the checkpoint; omit them when resuming");auto previous=load(model_path);tok.emplace(std::move(previous.tokenizer));model.emplace(std::move(previous.model));}
    else{
        constexpr std::size_t tokenizer_qa_repeat_cap=8;
        auto tokenizer_text=train_lines;
        std::vector<std::pair<std::string,std::string>> unique_qa;
        std::vector<std::size_t> qa_occurrences;
        for(const auto&qa:qa_train){
            const auto entry=std::find(unique_qa.begin(),unique_qa.end(),qa);
            if(entry==unique_qa.end()){unique_qa.push_back(qa);qa_occurrences.push_back(1);}
            else ++qa_occurrences[static_cast<std::size_t>(entry-unique_qa.begin())];
        }
        std::vector<std::size_t> tokenizer_copies(unique_qa.size(),0);
        for(const auto&qa:qa_train){
            const auto i=static_cast<std::size_t>(std::find(unique_qa.begin(),unique_qa.end(),qa)-unique_qa.begin());
            if(tokenizer_copies[i]<std::min(qa_occurrences[i],tokenizer_qa_repeat_cap)){
                tokenizer_text.push_back(qa.first);
                tokenizer_text.push_back(qa.second);
                ++tokenizer_copies[i];
            }
        }
        tok.emplace(TextTokenizer::train(tokenizer_text,vocab));
        model.emplace(tok->vocabularySize(),c);
    }
    auto training=sequences(*tok,tr,model->contextLength()),validation=sequences(*tok,va,model->contextLength());auto supervised=qaSequences(*tok,qa_train,model->contextLength());training.insert(training.end(),std::make_move_iterator(supervised.begin()),std::make_move_iterator(supervised.end()));if(training.empty()||validation.empty())throw std::runtime_error("Train and validation corpora must contain non-empty lines");
    std::vector<std::size_t> training_order(training.size());std::iota(training_order.begin(),training_order.end(),0);
    for(std::size_t e=0;e<epochs;++e){std::iota(training_order.begin(),training_order.end(),0);model->shuffleSequenceIndices(training_order);double train_loss=0;std::size_t train_tokens=0;for(std::size_t i=0;i<training_order.size();i+=batch){std::vector<LanguageSequence>b;b.reserve(std::min(batch,training_order.size()-i));std::size_t batch_tokens=0;for(std::size_t j=i;j<std::min(training_order.size(),i+batch);++j){b.push_back(training[training_order[j]]);const auto&seq=b.back();if(seq.loss_mask.empty())batch_tokens+=seq.target.size();else batch_tokens+=static_cast<std::size_t>(std::count(seq.loss_mask.begin(),seq.loss_mask.end(),1));}train_loss+=model->trainBatch(b)*batch_tokens;train_tokens+=batch_tokens;}auto m=model->evaluate(validation);std::cout<<"epoch="<<e+1<<" updates="<<model->updateCount()<<" parameters="<<model->parameterCount()<<" train_cross_entropy="<<(train_tokens?train_loss/train_tokens:0)<<" validation_cross_entropy="<<m.cross_entropy<<" validation_perplexity="<<m.perplexity<<" validation_tokens="<<m.token_count<<"\n";save(model_path,*tok,*model);}return 0;
}else if(command=="generate"){
    std::string path,prompt;std::size_t tokens=64;double temp=0.8,top_p=1;int top_k=0;std::uint32_t seed=1234;for(int i=3;i<argc;++i){std::string a=argv[i];if(a=="--model")path=value(i,argc,argv,a);else if(a=="--prompt")prompt=value(i,argc,argv,a);else if(a=="--tokens")tokens=std::stoul(value(i,argc,argv,a));else if(a=="--temperature")temp=number(value(i,argc,argv,a),a);else if(a=="--top-k")top_k=std::stoi(value(i,argc,argv,a));else if(a=="--top-p")top_p=number(value(i,argc,argv,a),a);else if(a=="--seed")seed=std::stoul(value(i,argc,argv,a));else throw std::invalid_argument("Unknown option: "+a);}if(path.empty()||prompt.empty()||temp<0||top_k<0||top_p<=0||top_p>1)throw std::invalid_argument("generate requires model, prompt and valid decoding options");auto loaded=load(path);auto ids=loaded.tokenizer.encode(prompt);ids.insert(ids.begin(),TextTokenizer::bosId);std::mt19937 rng(seed);std::vector<std::uint32_t> generated;for(std::size_t i=0;i<tokens;++i){auto next=sample(loaded.model.nextTokenLogits(ids),temp,top_k,top_p,rng).front();if(next==TextTokenizer::eosId)break;ids.push_back(next);generated.push_back(next);}std::cout<<prompt<<loaded.tokenizer.decode(generated)<<"\n";return 0;
}else if(command=="evaluate"){
    std::string path,text_path,qa_path;
    std::size_t weight_precision=0;
    std::vector<std::string> qa_labels;
    for(int i=3;i<argc;++i){
        std::string a=argv[i];
        if(a=="--model")path=value(i,argc,argv,a);
        else if(a=="--text")text_path=value(i,argc,argv,a);
        else if(a=="--qa")qa_path=value(i,argc,argv,a);
        else if(a=="--qa-label")qa_labels.push_back(value(i,argc,argv,a));
        else if(a=="--weight-precision"){const auto precision=value(i,argc,argv,a);if(precision=="int8")weight_precision=8;else if(precision=="int16")weight_precision=16;else throw std::invalid_argument("--weight-precision must be int8 or int16");}
        else throw std::invalid_argument("Unknown option: "+a);
    }
    if(path.empty()||(text_path.empty()&&qa_path.empty()))throw std::invalid_argument("evaluate requires --model and at least one of --text or --qa");
    if(!qa_labels.empty()&&(qa_path.empty()||qa_labels.size()<2))throw std::invalid_argument("--qa-label requires --qa and at least two candidate labels");
    std::unordered_set<std::string> unique_labels;
    for(const auto&label:qa_labels){
        const auto canonical=canonicalAnswer(label);
        if(canonical.empty()||!unique_labels.insert(canonical).second)
            throw std::invalid_argument("QA candidate labels must be non-empty and unique");
    }
    auto loaded=load(path);
    if(weight_precision){const auto quantization=loaded.model.quantizeWeights(weight_precision);std::cout<<"weight_precision_bits="<<quantization.bits<<" weight_max_abs_error="<<quantization.maximum_absolute_error<<" weight_mean_abs_error="<<quantization.mean_absolute_error<<"\n";}
    if(!text_path.empty()){
        auto data=sequences(loaded.tokenizer,readText(text_path),loaded.model.contextLength());
        if(data.empty())throw std::runtime_error("Text evaluation corpus contains no non-empty lines");
        auto m=loaded.model.evaluate(data);
        std::cout<<"cross_entropy="<<m.cross_entropy<<" perplexity="<<m.perplexity<<" tokens="<<m.token_count<<"\n";
    }
    if(!qa_path.empty()){
        auto pairs=lines(readText(qa_path));
        if(pairs.empty())throw std::runtime_error("QA evaluation corpus contains no non-empty rows");
        std::size_t correct=0,total=0,generated_tokens=0,closed_correct=0;
        double total_f1=0,generation_seconds=0,closed_seconds=0;
        for(const auto&pair:pairs){
            auto split=pair.find(" | ");
            if(split==std::string::npos)throw std::runtime_error("QA rows must use: prompt | reference answer");
            const auto prompt=pair.substr(0,split),reference=pair.substr(split+3);
            if(!qa_labels.empty()&&std::none_of(qa_labels.begin(),qa_labels.end(),[&](const auto&label){return canonicalAnswer(label)==canonicalAnswer(reference);}))
                throw std::runtime_error("QA reference is not included in the closed-set candidate labels");
            const auto generation_start=std::chrono::steady_clock::now();
            const auto generated=greedyCompletion(loaded.tokenizer,loaded.model,prompt,loaded.model.contextLength());
            generation_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-generation_start).count();
            if(canonicalAnswer(generated.text)==canonicalAnswer(reference))++correct;
            total_f1+=tokenF1(generated.text,reference);
            generated_tokens+=generated.token_count;
            if(!qa_labels.empty()){
                const auto closed_start=std::chrono::steady_clock::now();
                auto best=qa_labels.front();
                double best_score=continuationLogProbability(loaded.tokenizer,loaded.model,prompt,best);
                for(std::size_t i=1;i<qa_labels.size();++i){
                    const double score=continuationLogProbability(loaded.tokenizer,loaded.model,prompt,qa_labels[i]);
                    if(score>best_score){best_score=score;best=qa_labels[i];}
                }
                closed_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-closed_start).count();
                if(canonicalAnswer(best)==canonicalAnswer(reference))++closed_correct;
            }
            ++total;
        }
        std::cout<<"qa_exact_match="<<static_cast<double>(correct)/total
                 <<" qa_token_f1="<<total_f1/total<<" correct="<<correct<<" total="<<total
                 <<" generated_tokens="<<generated_tokens
                 <<" qa_generation_seconds="<<generation_seconds
                 <<" qa_generation_ms_per_example="<<(generation_seconds*1000.0/total)
                 <<" qa_generation_tokens_per_second="<<(generation_seconds>0?generated_tokens/generation_seconds:0);
        if(!qa_labels.empty())
            std::cout<<" qa_closed_set_accuracy="<<static_cast<double>(closed_correct)/total
                     <<" qa_closed_set_correct="<<closed_correct<<" qa_closed_set_total="<<total
                     <<" qa_closed_set_seconds="<<closed_seconds;
        std::cout<<"\n";
    }
    return 0;
}usage();return 2;}catch(const std::exception&e){std::cerr<<"Language runtime error: "<<e.what()<<"\n";return 1;}}
