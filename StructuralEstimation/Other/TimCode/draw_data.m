(* ::Package:: *)

SeedRandom[3]
(* load the Emax functions *)
Get[OutputDir<>"\settings.mx"]
Get[OutputDir<>"\\Emax"] 

GMult[a_,b_]:=(
prod=If[VectorQ[a],b.a,b*a];
Return[prod];
)

\[CapitalPhi][c_]:=CDF[NormalDistribution[0,1],c];
\[Phi][c_]:=PDF[NormalDistribution[0,1],c];


(* Draws data *)
Lz=Length[g1];
Ln=Length[Bn];
Lk=Length[Bk];
Ly=1;
TE[c_]:=ToExpression[c];

vars={"z","n","k","y"};


(* draws vectors of data *)
For[i=1,i<=Length[vars],i++,
Evaluate@TE["m"<>vars[[i]] ]=Table[0,{i,1,TE["L"<>vars[[i]] ]}];
Evaluate@TE["p"<>vars[[i]] ]=DiagonalMatrix[Table[3,{i,1,TE["L"<>vars[[i]] ]}]];
Evaluate@TE[vars[[i]] ]=RandomVariate[MultinormalDistribution[TE["m"<>vars[[i]]],TE["p"<>vars[[i]]]],M];
]
y=Flatten[y];

(* draws error terms for each period *)
For[i=1,i<=T,i++,
Evaluate@TE["eq"<>ToString[i]]= RandomVariate[MultinormalDistribution[{m\[Epsilon],m\[Eta]},{{p\[Eta],p\[Epsilon]\[Eta]},{p\[Epsilon]\[Eta],p\[Eta]}}], M];
Evaluate@TE["e"<>ToString[i]]=TE["eq"<>ToString[i]][[All,1]];
Evaluate@TE["q"<>ToString[i]]=TE["eq"<>ToString[i]][[All,2]];
]

(* sets the initial levels of experience *)


(* creates the decision functions for each period *)
h1=Table[0,{i,1,M,1}];
v1=GMult[g1,z]+GMult[g2,h1]-GMult[Bn,n]-GMult[Bk,k]+delta*(FEmaxT1[y,z,h1+1,n,k,g1,g2,Bk,Bn,stdxi,delta]-FEmaxT1[y,z,h1,n,k,g1,g2,Bk,Bn,stdxi,delta])+q1-e1;
d1=Table[ If[v1[[i]]>0,1,0],{i,1,M,1}];

h2=h1+d1;
v2=GMult[g1,z]+GMult[g2,h2]-GMult[Bn,n]-GMult[Bk,k]+delta*(FEmaxT[y,z,h2+1,n,k,g1,g2,Bk,Bn,stdxi]-FEmaxT[y,z,h2,n,k,g1,g2,Bk,Bn,stdxi])+q2-e2;
d2=Table[ If[v2[[i]]>0,1,0],{i,1,M,1}];

h3=h2+d2;
v3=GMult[g1,z]+GMult[g2,h3]-GMult[Bn,n]-GMult[Bk,k]+q3-e3;
d3=Table[ If[v3[[i]]>0,1,0],{i,1,M,1}];

w1=GMult[g1,z]+GMult[g2,h1]+q1;
w2=GMult[g1,z]+GMult[g2,h2]+q2;
w3=GMult[g1,z]+GMult[g2,h3]+q3;


 DumpSave[OutputDir<>"\\data_draw.mx",{y,d1,d2,d3,h1,h2,h3,w1,w2,w3,z,n,k}];
Print["Finished drawing "<>ToString[M]<> " observations."]
