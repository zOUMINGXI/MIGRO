#include <cstdint>
#include <cstring>
extern "C" {
int ans_encode(const int64_t* ids,int n,const int32_t* cdf,int v,uint8_t* out,int cap) {
 const uint64_t L=1u<<23;const int P=20;uint64_t s=L;int k=0;
 for(int i=n-1;i>=0;--i) {
  int64_t t=ids[i];if(t<0||t>=v)return -1;
  uint64_t a=cdf[t],f=cdf[t+1]-cdf[t];if(!f)return -2;
  while(s>=((L>>P)<<8)*f){if(k>=cap-4)return -3;out[k++]=s&255;s>>=8;}
  s=((s/f)<<P)+(s%f)+a;
 }
 for(int a=0,b=k-1;a<b;++a,--b){uint8_t t=out[a];out[a]=out[b];out[b]=t;}
 std::memmove(out+4,out,k);
 for(int j=0;j<4;++j)out[j]=(s>>(8*j))&255;
 return k+4;
}
int ans_decode(const uint8_t* in,int size,int n,const int32_t* cdf,int v,int64_t* out) {
 if(size<4)return -1;uint64_t s=0;for(int j=0;j<4;++j)s|=uint64_t(in[j])<<(8*j);
 int p=4;const int P=20;const uint64_t L=1u<<23;
 for(int i=0;i<n;++i){
  uint32_t x=s&((1u<<P)-1);int lo=0,hi=v;
  while(lo+1<hi){int m=(lo+hi)/2;if(uint32_t(cdf[m])<=x)lo=m;else hi=m;}
  out[i]=lo;uint64_t f=cdf[lo+1]-cdf[lo];s=f*(s>>P)+x-cdf[lo];
  while(s<L){if(p>=size)return -2;s=(s<<8)|in[p++];}
 }
 return (s==L&&p==size)?0:-3;
}
}

static inline bool push_ans(uint64_t& s,uint32_t a,uint32_t f,uint8_t* out,int& k,int cap) {
 const uint64_t L=1u<<23;const int P=20;
 while(s>=((L>>P)<<8)*uint64_t(f)){if(k>=cap-4)return false;out[k++]=s&255;s>>=8;}
 s=((s/f)<<P)+(s%f)+a;return true;
}
static inline int pull_ans(uint64_t& s,const int32_t* cdf,int v,const uint8_t* in,int size,int& p) {
 const int P=20;uint32_t x=s&((1u<<P)-1);int lo=0,hi=v;
 while(lo+1<hi){int m=(lo+hi)/2;if(uint32_t(cdf[m])<=x)lo=m;else hi=m;}
 s=uint64_t(cdf[lo+1]-cdf[lo])*(s>>P)+x-cdf[lo];
 while(s<(1u<<23)){if(p>=size)return -1;s=(s<<8)|in[p++];}
 return lo;
}
extern "C" {
int context_encode(const int64_t* ids,int n,const int32_t* base,int v,const int16_t* predictions,
 const int32_t* tables,const uint8_t* counts,int width,uint8_t* out,int cap) {
 uint64_t s=1u<<23;int k=0;
 for(int i=n-1;i>=0;--i){
  int prev=(i?int(ids[i-1]):v),token=ids[i],m=counts[prev],j=0;
  if(token<0||token>=v)return -1;
  const int16_t* pred=predictions+prev*width;const int32_t* cdf=tables+prev*(width+2);
  while(j<m&&pred[j]!=token)++j;
  if(j==m&&!push_ans(s,base[token],base[token+1]-base[token],out,k,cap))return -2;
  if(!push_ans(s,cdf[j],cdf[j+1]-cdf[j],out,k,cap))return -3;
 }
 for(int a=0,b=k-1;a<b;++a,--b){uint8_t t=out[a];out[a]=out[b];out[b]=t;}
 std::memmove(out+4,out,k);for(int j=0;j<4;++j)out[j]=(s>>(8*j))&255;return k+4;
}
int context_decode(const uint8_t* in,int size,int n,const int32_t* base,int v,const int16_t* predictions,
 const int32_t* tables,const uint8_t* counts,int width,int64_t* out) {
 if(size<4)return -1;uint64_t s=0;for(int j=0;j<4;++j)s|=uint64_t(in[j])<<(8*j);
 int p=4,prev=v;
 for(int i=0;i<n;++i){
  int m=counts[prev];int j=pull_ans(s,tables+prev*(width+2),m+1,in,size,p);if(j<0)return -2;
  int token=j==m?pull_ans(s,base,v,in,size,p):predictions[prev*width+j];if(token<0)return -3;
  out[i]=token;prev=token;
 }
 return(s==(1u<<23)&&p==size)?0:-4;
}
}
