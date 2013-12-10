(* ::Package:: *)

(* Defines the directories *)
BaseDir="C:\\Users\\Tim\\Dropbox\\Current Projects\\mathematica\\Tim\\dynamic_model";
CodeDir=BaseDir<>"\\code";
DataDir=BaseDir<>"\\data";
FigDir=BaseDir<>"\\figures";
OutputDir=BaseDir<>"\\output";
TableDir=BaseDir<>"\\tables";

Get[OutputDir<>"\\Emax.mx"];

\[CapitalPhi][c_]:=CDF[NormalDistribution[0,1],c];
\[Phi][c_]:=PDF[NormalDistribution[0,1],c];

GMult[a_,b_]:=(
prod=If[VectorQ[a],b.a,b*a];
Return[prod];
)

loglik[y_,d1_,d2_,d3_,h1_,h2_,h3_,w1_,w2_,w3_,z_,n_,k_,M_,\[Gamma]1_,\[Gamma]2_,\[Beta]n_,\[Beta]k_,\[Delta]_,\[Rho]_,\[Sigma]\[Epsilon]_,\[Sigma]\[Eta]_]:=(
minlog=10^-2000;
(* defines the wage error term *)
\[Omega]1=w1-GMult[\[Gamma]1,z]-GMult[\[Gamma]2,h1];
\[Omega]2=w2-GMult[\[Gamma]1,z]-GMult[\[Gamma]2,h2];
\[Omega]3=w3-GMult[\[Gamma]1,z]-GMult[\[Gamma]2,h3];

(* defines the observed part of the choice term *)
\[Xi]1=GMult[\[Gamma]1,z]+GMult[\[Gamma]2,h1]-GMult[\[Beta]n,n]-GMult[\[Beta]k,k]+\[Delta]*(FEmaxT1[y,z,h1+1,n,k,\[Gamma]1,\[Gamma]2,\[Beta]k,\[Beta]n,\[Sigma]\[Epsilon],\[Delta]]-FEmaxT1[y,z,h1,n,k,\[Gamma]1,\[Gamma]2,\[Beta]k,\[Beta]n,\[Sigma]\[Epsilon],\[Delta]]);

\[Xi]2=GMult[\[Gamma]1,z]+GMult[\[Gamma]2,h2]-GMult[\[Beta]n,n]-GMult[\[Beta]k,k]+\[Delta]*(FEmaxT[y,z,h2+1,n,k,\[Gamma]1,\[Gamma]2,\[Beta]k,\[Beta]n,\[Sigma]\[Epsilon]]-FEmaxT[y,z,h2,n,k,\[Gamma]1,\[Gamma]2,\[Beta]k,\[Beta]n,\[Sigma]\[Epsilon]]);

\[Xi]3=GMult[\[Gamma]1,z]+GMult[\[Gamma]2,h3]-GMult[\[Beta]n,n]-GMult[\[Beta]k,k];

(* defines the log-likelihood *)
\[ScriptCapitalL]=-Total[ Table[d1[[i]]*Log[Max[(  (1-\[CapitalPhi][(-\[Xi]1[[i]]-\[Rho]*\[Sigma]\[Epsilon]/\[Sigma]\[Eta]*\[Omega]1[[i]])/(\[Sigma]\[Epsilon]*(1-\[Rho]^2)^.5)])*1/\[Sigma]\[Eta] \[Phi][\[Omega]1[[i]]/\[Sigma]\[Eta]] ),minlog]]+(1-d1[[i]])*Log[Max[\[CapitalPhi][-\[Xi]1[[i]]/\[Sigma]\[Epsilon]],minlog]]
+d2[[i]]*Log[Max[(  (1-\[CapitalPhi][(-\[Xi]2[[i]]-\[Rho]*\[Sigma]\[Epsilon]/\[Sigma]\[Eta]*\[Omega]2[[i]])/(\[Sigma]\[Epsilon]*(1-\[Rho]^2)^.5)])*1/\[Sigma]\[Eta] \[Phi][\[Omega]2[[i]]/\[Sigma]\[Eta]] ),minlog]]+(1-d2[[i]])*Log[Max[\[CapitalPhi][-\[Xi]2[[i]]/\[Sigma]\[Epsilon]],minlog]]
+d3[[i]]*Log[Max[(  (1-\[CapitalPhi][(-\[Xi]3[[i]]-\[Rho]*\[Sigma]\[Epsilon]/\[Sigma]\[Eta]*\[Omega]3[[i]])/(\[Sigma]\[Epsilon]*(1-\[Rho]^2)^.5)])*1/\[Sigma]\[Eta] \[Phi][\[Omega]3[[i]]/\[Sigma]\[Eta]] ),minlog]]+(1-d3[[i]])*Log[Max[\[CapitalPhi][-\[Xi]3[[i]]/\[Sigma]\[Epsilon]],minlog]]
,{i,1,M,1}]

];
Return[\[ScriptCapitalL]];
)

DumpSave[OutputDir<>"\\likelihood.mx",loglik];
Print["Finished reading likelihood function."];
