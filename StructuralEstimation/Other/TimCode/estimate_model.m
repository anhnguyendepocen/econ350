(* ::Package:: *)

<<Ziena`
SeedRandom[10]
(* Defines the directories *)
BaseDir="C:\\Users\\Tim\\Dropbox\\Current Projects\\mathematica\\Tim\\dynamic_model";
CodeDir=BaseDir<>"\\code";
DataDir=BaseDir<>"\\data";
FigDir=BaseDir<>"\\figures";
OutputDir=BaseDir<>"\\output";
TableDir=BaseDir<>"\\tables";
Get[OutputDir<>"\\Emax.mx"];
Get[OutputDir<>"\\likelihood.mx"];
Get[OutputDir<>"\\data_draw.mx"];
Get[OutputDir<>"\\settings.mx"];



TE[c_]:=ToExpression[c];

param={"\[Gamma]1","\[Beta]n","\[Beta]k"}
pdata={"z","n","k"}

For[i=1,i<=Length[pdata],i++,
Dim=Dimensions[TE[pdata[[i]]]];
Evaluate@TE[param[[i]]<>"vec"]=Table[Symbol[param[[i]]<>ToString[j]],{j,Dim[[2]]}];
]

\[Gamma]1vec

(* Creates a vector of all the coefficients *)
coeffs=Join[\[Gamma]1vec,{\[Gamma]2},\[Beta]nvec,\[Beta]kvec,{\[Delta]}];

(* Creates vectors of the starting values for each coefficient *)
\[Gamma]1start=Table[0,{i,Length[\[Gamma]1vec]}];
\[Gamma]2start={0};
\[Beta]nstart=Table[0,{i,Length[\[Beta]nvec]}];
\[Beta]kstart=Table[0,{i,Length[\[Beta]kvec]}];
\[Delta]start={0};
coeffstart=Join[\[Gamma]1start,\[Gamma]2start,\[Beta]nstart,\[Beta]kstart,\[Delta]start];


varvec={\[Rho],\[Sigma]\[Epsilon],\[Sigma]\[Eta]};
varstart={.5,2,2};

allparam=Join[coeffs,varvec];
allstart=Join[coeffstart,varstart];

(* creates a vector of the lower and upper bounds on the coefficient variables *)
coefflo=Table[-Infinity,{i,1,Length[coeffs],1}]
coeffhi=Table[Infinity,{i,1,Length[coeffs],1}]

(* creates a vector of the lower and uppter bounds on the variance parameters *)
varlo={-0.999999,0.000001,0.000001}
varhi={0.999999,Infinity, Infinity}

lower=Join[coefflo,varlo]
upper=Join[coeffhi,varhi]

(* creates a function to resample data for the bootstrap loops *)
Resample[list_]:=list[[  Table [ Random[Integer,{1,Length[list]}],{Length[list]} ]]]

(* combines all of the data into a single table so that it can be resampled *)
data=Table[{  y[[i]],h1[[i]],h2[[i]],h3[[i]],d1[[i]],d2[[i]],d3[[i]],w1[[i]],w2[[i]],w3[[i]],z[[i]],n[[i]],k[[i]]  },{i,1,M,1}]

(* creates a list of the variables and their associated true values *)
vars={"y","h1","h2","h3","d1","d2","d3","w1","w2","w3","z","n","k"};
truevalues=Join[g1,{g2},Bn,Bk,{delta},{rho},{stdxi},{stdeta}];



(* estimates the parameters on the sample *)
estimate=KnitroMinimize[loglik[y,d1,d2,d3,h1,h2,h3,w1,w2,w3,z,n,k,M,\[Gamma]1vec,\[Gamma]2,\[Beta]nvec,\[Beta]kvec,\[Delta],\[Rho],\[Sigma]\[Epsilon],\[Sigma]\[Eta]],{lower,upper},allparam,KnitroOptionList->{{"outmode",1},{"outlev",6}},KnitroOptionList->{{"log_knml",1}},HonorBounds->True,Algorithm->"CG", Hessian->"SR1"]
estimate=estimate[[2]]
esttable=Table[{allparam[[i]],estimate[[i]],truevalues[[i]]},{i,1,Length[allparam]}];
Print[NumberForm[TableForm[Prepend[esttable,{"Param","Est.","True"}]], {4,3}, NumberSigns->{"-"," "}]]



(* runs the bootstrap loop *)

bootstrap=Table[0,{i,1,iter}];
total=0;
For[i=1,i<=iter,i++,
starttime=SessionTime[];
bdata=Resample[data];
For[j=1,j<=Length[vars],j++,
Clear[Evaluate["b"<>vars[[j]]] ];
Evaluate@TE["b"<>vars[[j]]]=bdata[[All,j]];
];
bootestimate=KnitroMinimize[loglik[by,bd1,bd2,bd3,bh1,bh2,bh3,bw1,bw2,bw3,bz,bn,bk,M,\[Gamma]1vec,\[Gamma]2,\[Beta]nvec,\[Beta]kvec,\[Delta],\[Rho],\[Sigma]\[Epsilon],\[Sigma]\[Eta]],{lower,upper},allparam,KnitroOptionList->{{"outmode",1},{"outlev",6}},KnitroOptionList->{{"log_knml",1}},HonorBounds->True,Algorithm->"CG", Hessian->"SR1"];
tempestimate=bootestimate[[2]];
bootstrap[[i]]=tempestimate;
itertime=Round[(SessionTime[]-starttime)/60,0.001];
total=total+itertime;
remainingtime=Round[(total/i)*(iter-i),0.001];
Print["Iteration "<>ToString[i]<>" took "<>ToString[itertime]<>" minutes. "<>ToString[iter-i]<> " remaining iterations. Est. time: " <>ToString[remainingtime]<>" minutes."  ];
];
bootstrap=DeleteCases[bootstrap,Null[[2]]]
finalests=Table[{allparam[[i]],estimate[[i]],Variance[bootstrap[[All,i]]]^.5,truevalues[[i]]},{i,1,Length[allparam]}];
Print[NumberForm[TableForm[Prepend[finalests,{"Param","Est.","SE","True"}]], {4,3}, NumberSigns->{"-"," "}]]

DumpSave[OutputDir<>"\\estimates.mx",{estimate,bootstrap,vars,iter,allparam}];
