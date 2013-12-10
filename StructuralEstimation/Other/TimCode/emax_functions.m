(* ::Package:: *)

BaseDir="C:\\Users\\Tim\\Dropbox\\Current Projects\\mathematica\\Tim\\dynamic_model";
CodeDir=BaseDir<>"\\code";
DataDir=BaseDir<>"\\data";
FigDir=BaseDir<>"\\figures";
OutputDir=BaseDir<>"\\output";
TableDir=BaseDir<>"\\tables";
Get[OutputDir<>"\settings.mx"]

(* This function is a general way to multiply a vector by an object that is either
either a scalar or a vector. The function recognizes which is which and 
applies the appropriate operation *)
GMult[a_,b_]:=(
prod=If[VectorQ[a],b.a,b*a];
Return[prod];
)

(* Simplifies the standard normal pdf and cdf for ease*)
\[CapitalPhi][c_]:=CDF[NormalDistribution[0,1],c];
\[Phi][c_]:=PDF[NormalDistribution[0,1],c];



(* Creates the emax function for the T period *)
FEmaxT[y_,z_,h_,n_,k_,\[Gamma]1_,\[Gamma]2_,\[Beta]k_,\[Beta]n_,\[Sigma]\[Xi]_]:=(
	(* Defines the observed part of the future periods decision *)
	\[Xi]=GMult[\[Gamma]1,z]+GMult[\[Gamma]2,h]-GMult[\[Beta]n,n]-GMult[\[Beta]k,k];
	(* Defines the function as in Keane, Todd, and Wolpin (2010) *)
	EmaxT=y+GMult[\[Beta]k,k]*\[CapitalPhi][-\[Xi]]+(GMult[\[Gamma]1,z]+GMult[\[Gamma]2,h]-GMult[\[Beta]n,n])*(1-\[CapitalPhi][-\[Xi]])+\[Sigma]\[Xi]*\[Phi][-\[Xi]];
	Return[EmaxT];
)


FEmaxT1[y_,z_,h_,n_,k_,\[Gamma]1_,\[Gamma]2_,\[Beta]k_,\[Beta]n_,\[Sigma]\[Xi]_,\[Delta]_]:=(
	(* Creates the EmaxT function for the next period if the agent does not work this period *)
	EmaxTh0=FEmaxT[y,z,h,n,k,\[Gamma]1,\[Gamma]2,\[Beta]k,\[Beta]n,\[Sigma]\[Xi]];
	(* Creates the EmaxT function for the next period if the agent works this period *)
	EmaxTh1=FEmaxT[y,z,h+1,n,k,\[Gamma]1,\[Gamma]2,\[Beta]k,\[Beta]n,\[Sigma]\[Xi]];
	(* Defines the observed component of the decision rule for the subsequent period *)
	\[Xi]=GMult[\[Gamma]1,z]+GMult[\[Gamma]2,h]-GMult[\[Beta]n,n]-GMult[\[Beta]k,k]+\[Delta]*(EmaxTh1-EmaxTh0);
	(* Defines the Emax function for the T-1 period *)
	EmaxT1=y+(GMult[\[Beta]k,k]+\[Delta]*EmaxTh0)*\[CapitalPhi][-\[Xi]]+(GMult[\[Gamma]1,z]+GMult[\[Gamma]2,h]-GMult[\[Beta]n,n]+\[Delta]*EmaxTh1)*(1-\[CapitalPhi][-\[Xi]])+\[Sigma]\[Xi]*\[Phi][-\[Xi]];
	Return[EmaxT1];
)

DumpSave[OutputDir<>"\\Emax.mx",{FEmaxT,FEmaxT1}]
Print["Finished Creating Emax functions."];
