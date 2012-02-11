% HW1, Q5 (ESL3.2)
% Dong-Bang Tsai
clear all;

N = 200;            % N of training set
alpha = (1-0.95)/2; % 95% confidence interval
sigma = 0.2;        % Gaussian random noise

% The training set
beta = randn(4,1);
x_points = rand(N,1); 
X = [ones(N,1),x_points,x_points.^2,x_points.^3]; 
y = X*beta + normrnd(0,sigma,N,1);
% Learned hypothesis
hat_beta = ((X'*X))\X'*y; 

% Plot array
N_plot = 50;
x_plot_points = [1/N_plot:1/N_plot:1]';
a = [ones(N_plot,1),x_plot_points,x_plot_points.^2,x_plot_points.^3];

% Compute the confidence band
hat_sigma = sqrt( 1/(N-4)*sum( (X*beta-y).^2 ) );
delta1 = norminv(1-alpha)*sqrt(diag(a*((X'*X)\a')))*hat_sigma;
delta2 = sqrt(diag(a*((X'*X)\a'))*chi2inv(1-2*alpha,4))*hat_sigma;

plot(x_points,y(:,1),'mx'); 
hold on;
plot(x_plot_points,a*beta,'k-');
plot(x_plot_points,a*beta + delta1,'r--'); 
plot(x_plot_points,a*beta - delta1,'r-.');
plot(x_plot_points,a*beta + delta2,'b--'); 
plot(x_plot_points,a*beta - delta2,'b-.');
xlabel('x'); ylabel('y');
legend('Training set','Linear least squares fitting','Method 1 Upper bound', 'Method 1 Lower bound', 'Method 2 Upper bound', 'Method 2 Lower bound');
hold off;
