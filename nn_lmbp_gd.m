clear 
clc

b = csvread('ext_features.csv');
%normc;

x = [min(b,[],1);max(b,[],1)];
M = bsxfun(@minus,b,x(1,:));
M = bsxfun(@rdivide,M,diff(x,1,1));

X = M(:,1:5);
Y = M(:,6);
YA = b(:,6);
trainY = Y(1:437,:);
ty = YA(1:437,:);
testY = YA(532:624,:);
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
trainX = I(1:437,:);
testX = I(438:624,:);
%defining the parameters of MFO

SearchAgents_no=30; % Number of search agents

Max_iteration=1000; % Maximum numbef of iterations

%Function_name='costfn'; % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)

% Load details of the cost function
[lb,ub,dim,fobj] = obj();

fprintf('MFO');

[Best_score,Best_pos,cg_curve]=MFO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj,ty,trainX);

%plotting the results

figure('Position',[284   214   660   290])

semilogy(cg_curve,'Color','b')
title('Convergence curve')
xlabel('Iteration');
ylabel('Best flame (score) obtained so far');

axis tight
grid off
box on
legend('MFO')

display(['The best solution obtained by MFO is : ', num2str(Best_pos)]);
display(['The best optimal value of the objective funciton found by MFO is : ', num2str(Best_score)]);

I = I';
Y = Y';                

net=feedforwardnet([],'traingdx');
net.trainParam.epochs = 200;
net.divideFcn= 'divideind'; % divide the data manually
net.divideParam.trainInd= 1:437; % training data indices 
net.divideParam.valInd= 438:531; % validation data indices 
net.divideParam.testInd= 532:624;  % testing data indices 

net.layers{1}.transferFcn = 'tansig';

net = configure(net,I,Y);

Best_pos = [0, Best_pos];
net = setwb(net,Best_pos);

net=train(net,I,Y);
view(net);
t = net(I);
p = perform(net,t,Y);
y = t';
y = y(532:624,:);
YP = ((y - 0.1)/0.8)*(max(testY) - min(testY)) + min(testY);
ct=0;
thres=[0.1 0.25 0.5 0.65 0.8];
for j=1:5
    ct=0;
for i=1:93
   if(abs(YP(i) - testY(i)) < thres(j))
       ct=ct+1;
   end
end
disp(ct/93);
end