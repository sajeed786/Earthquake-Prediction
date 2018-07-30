M = csvread('quake_flls_norm.csv');
X = M(2:23409,1:5);
Y = M(2:23409,6);
YA = M(2:23409,7);
trainY = Y(1:18726,:);
testY = YA(18727:23408,:);
d = size(X);
I = zeros(d(1),40);
for i = 1:(d(1))
    for j = 0:4
        I(i,8*j+1) = X(i,j+1);
        I(i,8*j+2) = 1/(1+exp(-X(i,j+1)));
        I(i,8*j+3) = sin(X(i,j+1));
        I(i,8*j+4) = sin(pi*X(i,j+1));
        I(i,8*j+5) = sin(2*pi*X(i,j+1));
        I(i,8*j+6) = cos(X(i,j+1)+0.3);
        I(i,8*j+7) = cos((pi*X(i,j+1))+0.3);
        I(i,8*j+8) = cos((2*pi*X(i,j+1))+0.3);
    end   
end
%division of expanded matrix into train and test sets
trainX = I(1:18726,:);
testX = I(18727:23408,:);
% finding weights according to least square regression and building the
% model
O = ones(size(trainY));
N = (O - trainY);
S = log(trainY ./ N);
T = inv(trainX' * trainX);
T1 = trainX' * S;
W = T * T1;
% Testing the model
Wsum = testX * W;
O = ones(size(testY));
Act = O ./ (O + exp(-Wsum));
YP = ((Act - 0.1)/0.8)*(max(testY) - min(testY)) + min(testY);
ct=0;
for i=1:4682
   if(abs(YP(i) - testY(i)) < 0.10)
       ct=ct+1;
   end
end
disp(ct/4682);