// Deterministic causal content mixture; no pretrained weights or hidden corpus.
// Probabilities depend only on previously decoded bytes and supplied past-visible scope.
#include <array>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
using namespace std;
static uint32_t mix(uint64_t x){x^=x>>33;x*=0xff51afd7ed558ccdULL;x^=x>>33;x*=0xc4ceb9fe1a85ec53ULL;x^=x>>33;return uint32_t(x);}
struct BitsOut { vector<uint8_t> v; int n=0; uint8_t b=0; void put(int bit){b=(b<<1)|bit;if(++n==8){v.push_back(b);b=0;n=0;}} void finish(){if(n){v.push_back(b<<(8-n));b=0;n=0;}} };
struct Encoder {uint64_t lo=0,hi=0xffffffffULL;uint32_t pending=0;BitsOut out;
 void emit(int b){out.put(b);while(pending){out.put(1-b);--pending;}}
 void bit(int b,uint32_t p1){uint64_t mid=lo+((hi-lo+1)*(65536-p1)>>16)-1;
  if(b)lo=mid+1;else hi=mid;
  while(true){if(hi<0x80000000ULL)emit(0);else if(lo>=0x80000000ULL){emit(1);lo-=0x80000000ULL;hi-=0x80000000ULL;}
   else if(lo>=0x40000000ULL && hi<0xc0000000ULL){pending++;lo-=0x40000000ULL;hi-=0x40000000ULL;}else break;
   lo<<=1;hi=(hi<<1)|1;}}
 vector<uint8_t> finish(){pending++;emit(lo<0x40000000ULL?0:1);out.finish();return out.v;}
};
struct Decoder {uint64_t lo=0,hi=0xffffffffULL,code=0;const vector<uint8_t>& data;size_t pos=0;
 int get(){int b=pos<data.size()*8?((data[pos>>3]>>(7-(pos&7)))&1):0;++pos;return b;}
 Decoder(const vector<uint8_t>&d):data(d){for(int i=0;i<32;i++)code=(code<<1)|get();}
 int bit(uint32_t p1){uint64_t mid=lo+((hi-lo+1)*(65536-p1)>>16)-1;int b=code>mid;
  if(b)lo=mid+1;else hi=mid;
  while(true){if(hi<0x80000000ULL){}else if(lo>=0x80000000ULL){lo-=0x80000000ULL;hi-=0x80000000ULL;code-=0x80000000ULL;}
   else if(lo>=0x40000000ULL && hi<0xc0000000ULL){lo-=0x40000000ULL;hi-=0x40000000ULL;code-=0x40000000ULL;}else break;
   lo<<=1;hi=(hi<<1)|1;code=(code<<1)|get();}return b;}
};
struct History {uint64_t bytes=0;uint32_t word=0,previous=0;void add(uint8_t c){bytes=(bytes<<8)|c;
 if((c>='a'&&c<='z')||(c>='A'&&c<='Z'))word=word*31+(c|32);else{if(word)previous=word;word=0;}}};
struct Model {
 static const int K=12;uint32_t mask;bool scopes;vector<array<uint16_t,2>> counts;
 array<array<uint32_t,K>,8> weight;array<uint32_t,K> ix,prob;
 History global;unordered_map<uint32_t,History> local;History* current=nullptr;uint32_t sid=0;
 Model(int bits,bool scoped):mask((1u<<bits)-1),scopes(scoped),counts(size_t(K)*(mask+1)) {
  for(auto &w:weight)w.fill((1u<<20)/K);}
 void record(uint32_t s){sid=s;current=&local[s];}
 uint32_t predict(uint32_t prefix,int bp){
  uint64_t s=uint64_t(sid)*0x9e3779b185ebca87ULL;auto &l=*current;
  array<uint64_t,K> h={0,global.bytes&255,global.bytes&65535,global.bytes&0xffffffffULL,global.bytes,
   uint64_t(global.word)*17+global.previous,uint64_t(global.word),
   s^(l.bytes&255),s^(l.bytes&65535),s^l.bytes,s^(uint64_t(l.word)*17+l.previous),s^l.word};
  uint64_t weighted=0,total=0;
  for(int k=0;k<K;k++){
   ix[k]=uint32_t(k*(mask+1))+(mix(h[k]*0x85ebca6b+prefix)&mask);
   auto c=counts[ix[k]];prob[k]=(uint32_t(c[1]+1)*65536)/(c[0]+c[1]+2);
   if(!scopes&&k>=7)continue;
   weighted+=uint64_t(weight[bp][k])*prob[k];total+=weight[bp][k];}
  return max(1u,min(65535u,uint32_t(weighted/total)));
 }
 void update(int b,int bp){
  uint64_t sum=0;array<uint32_t,K> next;
  for(int k=0;k<K;k++){
   auto &c=counts[ix[k]];if(c[0]+c[1]>=250){c[0]=(c[0]+1)/2;c[1]=(c[1]+1)/2;}c[b]++;
   uint32_t p=b?prob[k]:65536-prob[k];
   next[k]=max(1u,uint32_t((uint64_t(weight[bp][k])*p)>>16));
   if(scopes||k<7)sum+=next[k];
  }
  for(int k=0;k<K;k++)weight[bp][k]=(!scopes&&k>=7)?0:uint32_t(uint64_t(next[k])*(1u<<20)/sum)+32;
 }
 void add(uint8_t c){global.add(c);current->add(c);}
};
uint32_t read32(istream &f){uint32_t x;f.read((char*)&x,4);if(!f)throw runtime_error("truncated");return x;}
void write32(ostream&f,uint32_t x){f.write((char*)&x,4);}
int main(int argc,char**argv){
 try{
 if(argc!=6)throw runtime_error("encode/decode input output table_bits scoped");
 string mode=argv[1];ifstream in(argv[2],ios::binary);if(!in)throw runtime_error("input");
 int bits=stoi(argv[4]),scoped=stoi(argv[5]);Model model(bits,scoped);ofstream out(argv[3],ios::binary);
 uint32_t n=read32(in);vector<uint32_t> ids(n),lens(n);for(uint32_t i=0;i<n;i++){ids[i]=read32(in);lens[i]=read32(in);}
 write32(out,n);for(uint32_t i=0;i<n;i++){write32(out,ids[i]);write32(out,lens[i]);}
 if(mode=="encode"){
  Encoder coder;
  for(uint32_t i=0;i<n;i++){model.record(ids[i]);for(uint32_t j=0;j<lens[i];j++){
   uint8_t c;in.read((char*)&c,1);if(!in)throw runtime_error("data");uint32_t prefix=1;
   for(int bp=0;bp<8;bp++){int b=(c>>(7-bp))&1;auto p=model.predict(prefix,bp);coder.bit(b,p);model.update(b,bp);prefix=prefix*2+b;}model.add(c);}}
  auto data=coder.finish();out.write((char*)data.data(),data.size());
 }else if(mode=="decode"){
  vector<uint8_t> data((istreambuf_iterator<char>(in)),istreambuf_iterator<char>());Decoder coder(data);
  for(uint32_t i=0;i<n;i++){model.record(ids[i]);for(uint32_t j=0;j<lens[i];j++){
   uint32_t prefix=1;for(int bp=0;bp<8;bp++){auto p=model.predict(prefix,bp);int b=coder.bit(p);model.update(b,bp);prefix=prefix*2+b;}
   uint8_t c=prefix&255;out.write((char*)&c,1);model.add(c);}}
 }else throw runtime_error("mode");return 0;
 }catch(const exception&e){fprintf(stderr,"%s\n",e.what());return 1;}
}
