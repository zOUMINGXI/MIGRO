#include "rd_ans_native.cpp"
extern "C" {
int csr_encode(const int64_t* ids,int n,const int32_t* base,int v,const int16_t* predictions,
 const int32_t* tables,const int32_t* offsets,const int32_t* cdf_offsets,const uint8_t* counts,uint8_t* out,int cap) {
 uint64_t s=1u<<23;int k=0;
 for(int i=n-1;i>=0;--i){
  int prev=(i?int(ids[i-1]):v),token=ids[i],m=counts[prev],j=0;
  if(token<0||token>=v)return -1;
  const int16_t* pred=predictions+offsets[prev];const int32_t* cdf=tables+cdf_offsets[prev];
  while(j<m&&pred[j]!=token)++j;
  if(j==m&&!push_ans(s,base[token],base[token+1]-base[token],out,k,cap))return -2;
  if(!push_ans(s,cdf[j],cdf[j+1]-cdf[j],out,k,cap))return -3;
 }
 for(int a=0,b=k-1;a<b;++a,--b){uint8_t t=out[a];out[a]=out[b];out[b]=t;}
 std::memmove(out+4,out,k);for(int j=0;j<4;++j)out[j]=(s>>(8*j))&255;return k+4;
}
int csr_decode(const uint8_t* in,int size,int n,const int32_t* base,int v,const int16_t* predictions,
 const int32_t* tables,const int32_t* offsets,const int32_t* cdf_offsets,const uint8_t* counts,int64_t* out) {
 if(size<4)return -1;uint64_t s=0;for(int j=0;j<4;++j)s|=uint64_t(in[j])<<(8*j);
 int p=4,prev=v;
 for(int i=0;i<n;++i){
  int m=counts[prev];int j=pull_ans(s,tables+cdf_offsets[prev],m+1,in,size,p);if(j<0)return -2;
  int token=j==m?pull_ans(s,base,v,in,size,p):predictions[offsets[prev]+j];if(token<0)return -3;
  out[i]=token;prev=token;
 }
 return(s==(1u<<23)&&p==size)?0:-4;
}
}