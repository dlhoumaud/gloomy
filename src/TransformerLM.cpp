#include "headers/TransformerLM.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace {
struct Node { double value, gradient = 0.0; int a = -1, b = -1; double da = 0.0, db = 0.0; };
struct Graph {
    std::size_t precision_bits = 0;
    std::vector<Node> nodes;
    double reduced(double value) const { return precision_bits == 32 ? static_cast<double>(static_cast<float>(value)) : value; }
    int constant(double x) { nodes.push_back({reduced(x)}); return static_cast<int>(nodes.size()-1); }
    int unary(int a, double value, double derivative) { nodes.push_back({reduced(value),0,a,-1,derivative,0}); return static_cast<int>(nodes.size()-1); }
    int binary(int a,int b,double value,double da,double db) { nodes.push_back({reduced(value),0,a,b,da,db}); return static_cast<int>(nodes.size()-1); }
    void backward(int root) { nodes[root].gradient=1.0; for(int i=root;i>=0;--i){auto& n=nodes[i]; if(n.a>=0)nodes[n.a].gradient+=n.gradient*n.da; if(n.b>=0)nodes[n.b].gradient+=n.gradient*n.db;} }
};
struct V { Graph* g; int i; double val() const { return g->nodes[i].value; } };
V add(V a,V b){return {a.g,a.g->binary(a.i,b.i,a.val()+b.val(),1,1)};}
V sub(V a,V b){return {a.g,a.g->binary(a.i,b.i,a.val()-b.val(),1,-1)};}
V mul(V a,V b){return {a.g,a.g->binary(a.i,b.i,a.val()*b.val(),b.val(),a.val())};}
V scale(V a,double x){return {a.g,a.g->unary(a.i,a.val()*x,x)};}
V expv(V a){const double x=std::exp(a.val());return {a.g,a.g->unary(a.i,x,x)};}
V tanhv(V a){const double x=std::tanh(a.val());return {a.g,a.g->unary(a.i,x,1-x*x)};}
V constant(Graph& g,double x){return {&g,g.constant(x)};}
using Vec=std::vector<V>;
using Mat=std::vector<std::vector<V>>;
Vec linear(const Vec& x,const std::vector<V>& w,const std::vector<V>& bias,std::size_t in,std::size_t out){
    Vec y(out);
    const std::size_t bits=x.empty()?0:x[0].g->precision_bits;
    if(bits==8||bits==16){
        const std::int64_t maximum_level=(std::int64_t{1}<<(bits-1))-1;
        double max_input=0.0;
        for(auto value:x)max_input=std::max(max_input,std::abs(value.val()));
        const double input_scale=max_input==0.0?1.0:max_input/static_cast<double>(maximum_level);
        std::vector<std::int64_t> quantized_input(in);
        for(std::size_t i=0;i<in;++i)quantized_input[i]=static_cast<std::int64_t>(std::llround(x[i].val()/input_scale));
        for(std::size_t j=0;j<out;++j){
            double max_weight=0.0;
            for(std::size_t i=0;i<in;++i)max_weight=std::max(max_weight,std::abs(w[i*out+j].val()));
            const double weight_scale=max_weight==0.0?1.0:max_weight/static_cast<double>(maximum_level);
            std::int64_t accumulator=0;
            for(std::size_t i=0;i<in;++i){
                const auto quantized_weight=static_cast<std::int64_t>(std::llround(w[i*out+j].val()/weight_scale));
                accumulator+=quantized_input[i]*quantized_weight;
            }
            y[j]={x[0].g,x[0].g->constant(bias[j].val()+static_cast<double>(accumulator)*input_scale*weight_scale)};
        }
        return y;
    }
    if(bits==32){
        for(std::size_t j=0;j<out;++j){
            float value=static_cast<float>(bias[j].val());
            for(std::size_t i=0;i<in;++i)value+=static_cast<float>(x[i].val())*static_cast<float>(w[i*out+j].val());
            y[j]={x[0].g,x[0].g->constant(static_cast<double>(value))};
        }
        return y;
    }
    for(std::size_t j=0;j<out;++j){y[j]=bias[j];for(std::size_t i=0;i<in;++i)y[j]=add(y[j],mul(x[i],w[i*out+j]));}
    return y;
}
void quantizeActivation(Vec& values,std::size_t bits){
    if((bits!=8&&bits!=16)||values.empty())return;
    double maximum=0.0;
    for(auto value:values)maximum=std::max(maximum,std::abs(value.val()));
    const double level=static_cast<double>((std::uint32_t{1}<<(bits-1))-1);
    const double scale_value=maximum==0.0?1.0:maximum/level;
    for(auto& value:values){const double quantized=std::round(value.val()/scale_value)*scale_value;value={value.g,value.g->unary(value.i,quantized,1.0)};}
}
Vec norm(const Vec& x,const std::vector<V>& scalev,const std::vector<V>& bias){V mean=constant(*x[0].g,0);for(auto v:x)mean=add(mean,v);mean=scale(mean,1.0/x.size());V variance=constant(*x[0].g,0);for(auto v:x){V d=sub(v,mean);variance=add(variance,mul(d,d));}variance=scale(variance,1.0/x.size());V inv={variance.g,variance.g->unary(variance.i,1.0/std::sqrt(variance.val()+1e-5),-0.5*std::pow(variance.val()+1e-5,-1.5))};Vec y(x.size());for(std::size_t i=0;i<x.size();++i)y[i]=add(mul(mul(sub(x[i],mean),inv),scalev[i]),bias[i]);return y;}
V gelu(V x){ // tanh approximation
    V x3=mul(mul(x,x),x); V inner=scale(add(x,scale(x3,0.044715)),0.7978845608028654);
    return scale(mul(x,add(constant(*x.g,1),tanhv(inner))),0.5);
}
long double estimatedGraphNodes(std::size_t tokens,std::size_t vocabulary,const TransformerConfig& config){
    const long double length=tokens,dimension=config.model_dimension,layers=config.layer_count;
    return length*((2*dimension+4)*vocabulary+32*layers*dimension*dimension+16*dimension)
        +4*layers*dimension*length*length;
}
}

struct TransformerExecutor {
    static std::vector<std::vector<V>> allLogits(Graph& g,const TransformerLM& m,const std::vector<std::uint32_t>& ids,
                                 const std::vector<std::vector<V>>& p) {
        const auto D=m.model_config.model_dimension, H=m.model_config.attention_heads, HD=D/H;
        const auto L=std::min(ids.size(),m.model_config.context_length);
        std::vector<Vec> x(L,Vec(D));
        for(std::size_t t=0;t<L;++t) for(std::size_t d=0;d<D;++d)
            x[t][d]=add(p[m.token_embedding][ids[t]*D+d],p[m.position_embedding][t*D+d]);
        for(const auto& b:m.blocks){
            std::vector<Vec> q(L,Vec(D)),k(L,Vec(D)),v(L,Vec(D));
            for(std::size_t t=0;t<L;++t){
                auto n=norm(x[t],std::vector<V>(p[b.norm1_scale].begin(),p[b.norm1_scale].end()),std::vector<V>(p[b.norm1_bias].begin(),p[b.norm1_bias].end()));
                auto qkv=linear(n,p[b.qkv_weight],p[b.qkv_bias],D,3*D);
                for(std::size_t d=0;d<D;++d){q[t][d]=qkv[d];k[t][d]=qkv[D+d];v[t][d]=qkv[2*D+d];}
                if(m.inference_precision_bits==8||m.inference_precision_bits==16){quantizeActivation(q[t],m.inference_precision_bits);quantizeActivation(k[t],m.inference_precision_bits);quantizeActivation(v[t],m.inference_precision_bits);}
            }
            std::vector<Vec> attended(L,Vec(D));
            for(std::size_t t=0;t<L;++t) for(std::size_t h=0;h<H;++h){
                Vec scores; scores.reserve(t+1);
                for(std::size_t s=0;s<=t;++s){V z=constant(g,0);for(std::size_t d=0;d<HD;++d)z=add(z,mul(q[t][h*HD+d],k[s][h*HD+d]));scores.push_back(scale(z,1.0/std::sqrt(static_cast<double>(HD))));}
                double mx=-std::numeric_limits<double>::infinity();for(auto z:scores)mx=std::max(mx,z.val());Vec probs;V sum=constant(g,0);
                for(auto z:scores){probs.push_back(expv(sub(z,constant(g,mx))));sum=add(sum,probs.back());}
                for(auto& z:probs)z=mul(z,{sum.g,sum.g->unary(sum.i,1.0/sum.val(),-1.0/(sum.val()*sum.val()))});
                for(std::size_t d=0;d<HD;++d){V z=constant(g,0);for(std::size_t s=0;s<=t;++s)z=add(z,mul(probs[s],v[s][h*HD+d]));attended[t][h*HD+d]=z;}
            }
            if(m.inference_precision_bits==8||m.inference_precision_bits==16)for(auto& row:attended)quantizeActivation(row,m.inference_precision_bits);
            for(std::size_t t=0;t<L;++t){
                auto a=linear(attended[t],p[b.attention_weight],p[b.attention_bias],D,D);
                for(std::size_t d=0;d<D;++d)x[t][d]=add(x[t][d],a[d]);
                auto n=norm(x[t],p[b.norm2_scale],p[b.norm2_bias]);
                auto ff=linear(n,p[b.feed_forward_in_weight],p[b.feed_forward_in_bias],D,4*D);
                for(auto& z:ff)z=gelu(z);
                if(m.inference_precision_bits==8||m.inference_precision_bits==16)quantizeActivation(ff,m.inference_precision_bits);
                auto out=linear(ff,p[b.feed_forward_out_weight],p[b.feed_forward_out_bias],4*D,D);
                for(std::size_t d=0;d<D;++d)x[t][d]=add(x[t][d],out[d]);
            }
        }
        std::vector<std::vector<V>> result;result.reserve(L);
        for(const auto& row:x){auto last=norm(row,p[m.final_norm_scale],p[m.final_norm_bias]);result.push_back(linear(last,p[m.output_weight],p[m.output_bias],D,m.vocabulary_size));}
        return result;
    }
    static std::vector<V> logits(Graph&g,const TransformerLM&m,const std::vector<std::uint32_t>&ids,const std::vector<std::vector<V>>&p){return allLogits(g,m,ids,p).back();}
    static std::vector<std::vector<V>> leaves(Graph& g,const TransformerLM& m){std::vector<std::vector<V>> p;for(const auto& param:m.parameters){p.emplace_back();p.back().reserve(param.value.size());for(double x:param.value)p.back().push_back({&g,g.constant(x)});}return p;}
};

namespace {
template<class T> void writeRaw(std::ostream& s,const T& x){s.write(reinterpret_cast<const char*>(&x),sizeof(T));if(!s)throw std::runtime_error("Unable to write language checkpoint");}
template<class T> void readRaw(std::istream& s,T& x){s.read(reinterpret_cast<char*>(&x),sizeof(T));if(!s)throw std::runtime_error("Truncated language checkpoint");}
}

TransformerLM::TransformerLM(std::size_t vocab, TransformerConfig cfg)
    : vocabulary_size(vocab),model_config(cfg),token_embedding(0),position_embedding(0),final_norm_scale(0),final_norm_bias(0),output_weight(0),output_bias(0),random_generator(cfg.seed){
    if(vocab<259||vocab>1000000||cfg.context_length==0||cfg.context_length>2048||cfg.model_dimension<4||cfg.model_dimension>512||cfg.attention_heads==0||cfg.model_dimension%cfg.attention_heads||cfg.layer_count==0||cfg.layer_count>16||cfg.learning_rate<=0||!std::isfinite(cfg.learning_rate)||cfg.weight_decay<0||!std::isfinite(cfg.weight_decay))throw std::invalid_argument("Invalid language model configuration");
    const long double d=cfg.model_dimension;
    const long double estimated_parameters=2.0L*vocab*d+cfg.context_length*d+cfg.layer_count*(12*d*d+13*d)+2*d+vocab;
    if(estimated_parameters>2'000'000)throw std::invalid_argument("Language model exceeds the 2000000 parameter safety limit");
    initializeParameters();
}
std::size_t TransformerLM::addParameter(std::size_t size,bool decay,double initial){parameters.push_back({std::vector<double>(size,initial),std::vector<double>(size,0),std::vector<double>(size,0),decay});return parameters.size()-1;}
void TransformerLM::initializeParameters(){
    const auto d=model_config.model_dimension,v=vocabulary_size;std::normal_distribution<double> emb(0,0.02);
    token_embedding=addParameter(v*d,true);position_embedding=addParameter(model_config.context_length*d,true);
    for(auto index:{token_embedding,position_embedding})for(auto&x:parameters[index].value)x=emb(random_generator);
    for(std::size_t l=0;l<model_config.layer_count;++l){BlockParameters b;
        b.norm1_scale=addParameter(d,false,1);b.norm1_bias=addParameter(d,false);
        b.qkv_weight=addParameter(d*3*d,true);b.qkv_bias=addParameter(3*d,false);
        b.attention_weight=addParameter(d*d,true);b.attention_bias=addParameter(d,false);
        b.norm2_scale=addParameter(d,false,1);b.norm2_bias=addParameter(d,false);
        b.feed_forward_in_weight=addParameter(d*4*d,true);b.feed_forward_in_bias=addParameter(4*d,false);
        b.feed_forward_out_weight=addParameter(4*d*d,true);b.feed_forward_out_bias=addParameter(d,false);blocks.push_back(b);
        for(auto index:{b.qkv_weight,b.attention_weight,b.feed_forward_in_weight,b.feed_forward_out_weight}){const auto fan=std::sqrt(2.0/static_cast<double>(d+4*d));std::normal_distribution<double> init(0,fan);for(auto&x:parameters[index].value)x=init(random_generator);}
    }
    final_norm_scale=addParameter(d,false,1);final_norm_bias=addParameter(d,false);output_weight=addParameter(d*v,true);output_bias=addParameter(v,false);
    std::normal_distribution<double> out(0,std::sqrt(2.0/(d+v)));for(auto&x:parameters[output_weight].value)x=out(random_generator);
}

double TransformerLM::trainBatch(const std::vector<LanguageSequence>& batch){
    if(inference_precision_bits!=0)throw std::logic_error("Reduced precision mode is inference-only");
    if(batch.empty()) throw std::invalid_argument("Language training batch is empty");
    long double graph_budget=0;
    for(const auto& sequence:batch)graph_budget+=estimatedGraphNodes(sequence.input.size(),vocabulary_size,model_config);
    if(graph_budget>4'000'000)throw std::invalid_argument("Training batch exceeds the 4000000-node CPU autodiff safety budget; reduce batch size, context or model dimension");
    Graph g;auto p=TransformerExecutor::leaves(g,*this);V loss=constant(g,0);std::size_t count=0;
    for(const auto& seq:batch){if(seq.input.empty()||seq.input.size()!=seq.target.size())throw std::invalid_argument("Language sequence input/target sizes differ or are empty");if(!seq.loss_mask.empty()&&seq.loss_mask.size()!=seq.target.size())throw std::invalid_argument("Language loss mask size differs from target size");for(auto mask:seq.loss_mask)if(mask>1)throw std::invalid_argument("Language loss mask values must be zero or one");if(seq.input.size()>contextLength())throw std::invalid_argument("Language sequence exceeds context length");
        for(auto id:seq.input) if(id>=vocabulary_size) throw std::invalid_argument("Language token id outside vocabulary");
        for(auto id:seq.target) if(id>=vocabulary_size) throw std::invalid_argument("Language target id outside vocabulary");
        auto rows=TransformerExecutor::allLogits(g,*this,seq.input,p);
        for(std::size_t t=0;t<seq.target.size();++t){if(!seq.loss_mask.empty()&&!seq.loss_mask[t])continue;auto&logits=rows[t];double mx=-std::numeric_limits<double>::infinity();for(auto x:logits)mx=std::max(mx,x.val());V sum=constant(g,0);std::vector<V> ex;for(auto x:logits){ex.push_back(expv(sub(x,constant(g,mx))));sum=add(sum,ex.back());}V lp=sub(sub(logits[seq.target[t]],constant(g,mx)),{sum.g,sum.g->unary(sum.i,std::log(sum.val()),1.0/sum.val())});loss=sub(loss,lp);++count;}}
    if(!count) throw std::invalid_argument("No language tokens in batch");
    for (const auto& node : g.nodes) {
        if (!std::isfinite(node.value))
            throw std::runtime_error("Non-finite value in Transformer training graph");
    }
    loss=scale(loss,1.0/count);g.backward(loss.i);
    const double beta1=.9,beta2=.999,eps=1e-8;
    // Each parameter leaf is created in the same order as the flattened nodes.
    long double norm_squared=0;std::size_t node=0;for(const auto&param:parameters)for(std::size_t i=0;i<param.value.size();++i){const double grad=g.nodes[node++].gradient;if(!std::isfinite(grad))throw std::runtime_error("Non-finite gradient in Transformer training graph");norm_squared+=static_cast<long double>(grad)*grad;}
    if(!std::isfinite(norm_squared))throw std::runtime_error("Non-finite gradient norm in Transformer training graph");
    const double clip=norm_squared>1.0L?static_cast<double>(1.0L/std::sqrt(norm_squared)):1.0;
    if(update_count==std::numeric_limits<std::uint64_t>::max())
        throw std::runtime_error("Transformer update counter is exhausted");
    const auto step=update_count+1;
    node=0;
    for(const auto&param:parameters)for(std::size_t i=0;i<param.value.size();++i){
        const double grad=g.nodes[node++].gradient*clip;
        const double first=beta1*param.first_moment[i]+(1-beta1)*grad;
        const double second=beta2*param.second_moment[i]+(1-beta2)*grad*grad;
        const double corrected_first=first/(1-std::pow(beta1,static_cast<double>(step)));
        const double corrected_second=second/(1-std::pow(beta2,static_cast<double>(step)));
        const double value=param.value[i]-model_config.learning_rate*(corrected_first/(std::sqrt(corrected_second)+eps)+(param.decay?model_config.weight_decay*param.value[i]:0));
        if(!std::isfinite(first)||!std::isfinite(second)||!std::isfinite(value))
            throw std::runtime_error("Non-finite Transformer parameter update rejected");
    }
    update_count=step;
    node=0;for(auto&param:parameters)for(std::size_t i=0;i<param.value.size();++i){double grad=g.nodes[node++].gradient*clip;param.first_moment[i]=beta1*param.first_moment[i]+(1-beta1)*grad;param.second_moment[i]=beta2*param.second_moment[i]+(1-beta2)*grad*grad;const double m=param.first_moment[i]/(1-std::pow(beta1,static_cast<double>(step))),v=param.second_moment[i]/(1-std::pow(beta2,static_cast<double>(step)));param.value[i]-=model_config.learning_rate*(m/(std::sqrt(v)+eps)+(param.decay?model_config.weight_decay*param.value[i]:0));}
    for (const auto& param : parameters) {
        for (std::size_t i = 0; i < param.value.size(); ++i) {
            if (!std::isfinite(param.value[i]) || !std::isfinite(param.first_moment[i]) ||
                !std::isfinite(param.second_moment[i]))
                throw std::runtime_error("Non-finite parameter or Adam state after Transformer update");
        }
    }
    return loss.val();
}

double TransformerLM::gradientCheck(const std::vector<LanguageSequence>& batch, double epsilon) const {
    if(inference_precision_bits!=0)throw std::logic_error("Gradient checks require float64 mode");
    if (batch.empty() || !std::isfinite(epsilon) || epsilon <= 0.0)
        throw std::invalid_argument("Gradient check requires a batch and a positive finite epsilon");
    Graph graph;
    auto parameter_nodes = TransformerExecutor::leaves(graph, *this);
    V loss = constant(graph, 0.0);
    std::size_t token_count = 0;
    for (const auto& sequence : batch) {
        if (sequence.input.empty() || sequence.input.size() != sequence.target.size() ||
            sequence.input.size() > contextLength() ||
            (!sequence.loss_mask.empty() && sequence.loss_mask.size() != sequence.target.size()))
            throw std::invalid_argument("Invalid sequence passed to Transformer gradient check");
        if (!sequence.loss_mask.empty())
            for (auto mask : sequence.loss_mask)
                if (mask > 1) throw std::invalid_argument("Invalid loss mask in Transformer gradient check");
        for (auto id : sequence.input)
            if (id >= vocabulary_size) throw std::invalid_argument("Invalid input token in Transformer gradient check");
        for (auto id : sequence.target)
            if (id >= vocabulary_size) throw std::invalid_argument("Invalid target token in Transformer gradient check");
        const auto rows = TransformerExecutor::allLogits(graph, *this, sequence.input, parameter_nodes);
        for (std::size_t t = 0; t < sequence.target.size(); ++t) {
            if (!sequence.loss_mask.empty() && !sequence.loss_mask[t]) continue;
            const auto& logits = rows[t];
            double maximum = -std::numeric_limits<double>::infinity();
            for (auto value : logits) maximum = std::max(maximum, value.val());
            V sum = constant(graph, 0.0);
            std::vector<V> exponentials;
            exponentials.reserve(logits.size());
            for (auto value : logits) {
                exponentials.push_back(expv(sub(value, constant(graph, maximum))));
                sum = add(sum, exponentials.back());
            }
            V log_probability = sub(
                sub(logits[sequence.target[t]], constant(graph, maximum)),
                {sum.g, sum.g->unary(sum.i, std::log(sum.val()), 1.0 / sum.val())});
            loss = sub(loss, log_probability);
            ++token_count;
        }
    }
    if (!token_count) throw std::invalid_argument("Gradient check batch has no supervised tokens");
    loss = scale(loss, 1.0 / token_count);
    graph.backward(loss.i);

    auto loss_at = [&](const TransformerLM& model) {
        const auto metrics = model.evaluate(batch);
        if (!std::isfinite(metrics.cross_entropy))
            throw std::runtime_error("Non-finite loss during Transformer gradient check");
        return metrics.cross_entropy;
    };
    double maximum_relative_error = 0.0;
    std::size_t leaf = 0;
    for (std::size_t parameter_index = 0; parameter_index < parameters.size(); ++parameter_index) {
        const auto& parameter = parameters[parameter_index];
        const std::size_t probes[] = {0, parameter.value.size() / 2, parameter.value.size() - 1};
        for (std::size_t probe = 0; probe < 3; ++probe) {
            const auto index = probes[probe];
            const double analytic = graph.nodes[leaf + index].gradient;
            TransformerLM plus = *this;
            TransformerLM minus = *this;
            plus.parameters[parameter_index].value[index] += epsilon;
            minus.parameters[parameter_index].value[index] -= epsilon;
            const double numerical = (loss_at(plus) - loss_at(minus)) / (2.0 * epsilon);
            const double scale = std::max({1e-6, std::abs(analytic), std::abs(numerical)});
            maximum_relative_error = std::max(maximum_relative_error, std::abs(analytic - numerical) / scale);
        }
        leaf += parameter.value.size();
    }
    return maximum_relative_error;
}

WeightQuantizationMetrics TransformerLM::quantizeWeights(std::size_t bits) {
    if (bits != 8 && bits != 16)
        throw std::invalid_argument("Transformer weight quantization supports only 8 or 16 bits");
    const double maximum_level = static_cast<double>((std::uint32_t{1} << (bits - 1)) - 1);
    long double absolute_error_sum = 0.0;
    std::size_t value_count = 0;
    WeightQuantizationMetrics metrics;
    metrics.bits = bits;
    for (auto& parameter : parameters) {
        if (!parameter.decay) continue; // biases and normalization vectors stay float64
        double maximum_absolute_value = 0.0;
        for (double value : parameter.value)
            maximum_absolute_value = std::max(maximum_absolute_value, std::abs(value));
        const double scale = maximum_absolute_value == 0.0 ? 1.0 : maximum_absolute_value / maximum_level;
        for (double& value : parameter.value) {
            const double original = value;
            const double quantized = std::round(original / scale) * scale;
            const double error = std::abs(original - quantized);
            metrics.maximum_absolute_error = std::max(metrics.maximum_absolute_error, error);
            absolute_error_sum += error;
            ++value_count;
            value = quantized;
        }
    }
    if (value_count) metrics.mean_absolute_error = static_cast<double>(absolute_error_sum / value_count);
    return metrics;
}

void TransformerLM::setInferencePrecision(std::size_t bits) {
    if(bits!=0&&bits!=8&&bits!=16&&bits!=32)
        throw std::invalid_argument("Inference precision must be float64 (0), float32 (32), int8 or int16");
    inference_precision_bits=bits;
}

void TransformerLM::shuffleTrainingSequences(std::vector<LanguageSequence>& sequences){std::shuffle(sequences.begin(),sequences.end(),random_generator);}
void TransformerLM::shuffleSequenceIndices(std::vector<std::size_t>& indices){std::shuffle(indices.begin(),indices.end(),random_generator);}

LanguageMetrics TransformerLM::evaluate(const std::vector<LanguageSequence>& seqs)const{
    LanguageMetrics result;double total=0;for(const auto&s:seqs){if(s.input.size()!=s.target.size()||s.input.empty())continue;if(!s.loss_mask.empty()&&s.loss_mask.size()!=s.target.size())throw std::invalid_argument("Evaluation loss mask size differs from target size");if(s.input.size()>contextLength())throw std::invalid_argument("Evaluation sequence exceeds context length");if(estimatedGraphNodes(s.input.size(),vocabulary_size,model_config)>4'000'000)throw std::invalid_argument("Evaluation sequence exceeds the CPU autodiff safety budget");Graph g;g.precision_bits=inference_precision_bits;auto p=TransformerExecutor::leaves(g,*this);auto rows=TransformerExecutor::allLogits(g,*this,s.input,p);for(std::size_t t=0;t<s.input.size();++t){if(!s.loss_mask.empty()&&!s.loss_mask[t])continue;const auto&logits=rows[t];double mx=-INFINITY,sum=0;for(auto z:logits)mx=std::max(mx,z.val());for(auto z:logits)sum+=std::exp(z.val()-mx);if(s.target[t]>=logits.size())throw std::invalid_argument("Language target id outside vocabulary");total+=-(logits[s.target[t]].val()-mx-std::log(sum));++result.token_count;}}
    if(result.token_count){result.cross_entropy=total/result.token_count;result.perplexity=std::exp(std::min(80.0,result.cross_entropy));}return result;
}
std::vector<double> TransformerLM::nextTokenLogits(const std::vector<std::uint32_t>& context)const{if(context.empty())throw std::invalid_argument("Prompt must contain at least one token");const auto start=context.size()>contextLength()?context.size()-contextLength():0;std::vector<std::uint32_t> ids(context.begin()+start,context.end());for(auto id:ids)if(id>=vocabulary_size)throw std::invalid_argument("Prompt token id outside vocabulary");if(estimatedGraphNodes(ids.size(),vocabulary_size,model_config)>4'000'000)throw std::invalid_argument("Prompt exceeds the CPU autodiff safety budget");Graph g;g.precision_bits=inference_precision_bits;auto p=TransformerExecutor::leaves(g,*this);auto logits=TransformerExecutor::logits(g,*this,ids,p);std::vector<double> result;for(auto x:logits)result.push_back(x.val());return result;}
std::size_t TransformerLM::vocabularySize()const{return vocabulary_size;}std::size_t TransformerLM::parameterCount()const{std::size_t count=0;for(const auto&p:parameters)count+=p.value.size();return count;}std::size_t TransformerLM::contextLength()const{return model_config.context_length;}const TransformerConfig& TransformerLM::config()const{return model_config;}std::uint64_t TransformerLM::updateCount()const{return update_count;}
void TransformerLM::writeCheckpoint(std::ostream&s)const{const char magic[8]={'G','L','M','T','R','F','2','\0'};s.write(magic,8);const std::uint32_t ver=2;writeRaw(s,ver);writeRaw(s,vocabulary_size);writeRaw(s,model_config.context_length);writeRaw(s,model_config.model_dimension);writeRaw(s,model_config.attention_heads);writeRaw(s,model_config.layer_count);writeRaw(s,model_config.learning_rate);writeRaw(s,model_config.weight_decay);writeRaw(s,model_config.seed);writeRaw(s,update_count);const std::uint64_t n=parameters.size();writeRaw(s,n);for(const auto&p:parameters){const std::uint64_t z=p.value.size();writeRaw(s,z);writeRaw(s,p.decay);s.write(reinterpret_cast<const char*>(p.value.data()),z*sizeof(double));s.write(reinterpret_cast<const char*>(p.first_moment.data()),z*sizeof(double));s.write(reinterpret_cast<const char*>(p.second_moment.data()),z*sizeof(double));if(!s)throw std::runtime_error("Unable to write Transformer checkpoint");}std::ostringstream rng;rng<<random_generator;const std::string state=rng.str();const std::uint64_t state_size=state.size();writeRaw(s,state_size);s.write(state.data(),static_cast<std::streamsize>(state.size()));if(!s)throw std::runtime_error("Unable to serialize Transformer RNG state");}
TransformerLM TransformerLM::readCheckpoint(std::istream&s){char magic[8];s.read(magic,8);const char expected[8]={'G','L','M','T','R','F','2','\0'};if(!s||std::memcmp(magic,expected,8))throw std::runtime_error("Not a GLOOMY Transformer checkpoint");std::uint32_t ver,seed;std::uint64_t vocab,ctx,d,heads,layers,updates;double lr,wd;readRaw(s,ver);readRaw(s,vocab);readRaw(s,ctx);readRaw(s,d);readRaw(s,heads);readRaw(s,layers);readRaw(s,lr);readRaw(s,wd);readRaw(s,seed);readRaw(s,updates);if(ver!=2)throw std::runtime_error("Unsupported Transformer checkpoint version");TransformerConfig c;c.context_length=ctx;c.model_dimension=d;c.attention_heads=heads;c.layer_count=layers;c.learning_rate=lr;c.weight_decay=wd;c.seed=seed;TransformerLM m(vocab,c);std::uint64_t n;readRaw(s,n);if(n!=m.parameters.size())throw std::runtime_error("Transformer checkpoint parameter layout mismatch");for(auto&p:m.parameters){std::uint64_t z;bool decay;readRaw(s,z);readRaw(s,decay);if(z!=p.value.size()||decay!=p.decay)throw std::runtime_error("Transformer checkpoint tensor shape mismatch");s.read(reinterpret_cast<char*>(p.value.data()),z*sizeof(double));s.read(reinterpret_cast<char*>(p.first_moment.data()),z*sizeof(double));s.read(reinterpret_cast<char*>(p.second_moment.data()),z*sizeof(double));if(!s)throw std::runtime_error("Truncated Transformer checkpoint tensors");for(double x:p.value)if(!std::isfinite(x))throw std::runtime_error("Non-finite Transformer checkpoint value");for(double x:p.first_moment)if(!std::isfinite(x))throw std::runtime_error("Non-finite Adam first moment");for(double x:p.second_moment)if(!std::isfinite(x)||x<0)throw std::runtime_error("Invalid Adam second moment");}std::uint64_t state_size;readRaw(s,state_size);if(state_size>100000)throw std::runtime_error("Transformer RNG state exceeds size limit");std::string state(static_cast<std::size_t>(state_size),'\0');s.read(state.data(),static_cast<std::streamsize>(state.size()));if(!s)throw std::runtime_error("Truncated Transformer RNG state");std::istringstream rng(state);rng>>m.random_generator;if(!rng)throw std::runtime_error("Invalid Transformer RNG state");m.update_count=updates;return m;}
