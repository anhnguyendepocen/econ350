This version: 12/10/2013.

The data set Probit comes from the NLSY79. The data set is representative of the 1979 young
cohort (ages 14-22). It contains eleven variables and three periods of observation.

id: an observation numerator
weightr: a weight that induces representation of the 1979 oung cohort (ages 14-22)
year: a year indicator, 1996; 2002; 2008. The data contains three observation per individual
poverty*: a family poverty indicator
children: number of children in current household
hgc: highest school grade completed
afqt: a afqt score standarized to have mean 100 and standard deviation 15
single: single marital status indicator
separated: separated marital status indicator
divorced: divorced marital status indicator
widowed: widowed marital status indicator

The total of observations of pooled observations is 20, 210. 
The data set is in .csv format to ease use in any platform.

*Poverty is measured at family level. 
A non-farm family is poor if its income over the last 12 months 
is equal or less than $3140 + (1020*(N-1)), where N is the number of individuals in the family.
A farm family is poor if its income over the last 12 months 
is equal or less than $2690 + (860*(N-1)), where N is the number of individuals in the family.

Other: 
1. Use poverty as dependent variable. 
2. Use weightr as your frequency weights.
3. Try different specifications of the dependent variable. 
4. Make sure you include one specification per year that includes
children, hgc, afqt, single, separated, divorced, and widowed as independent variables.