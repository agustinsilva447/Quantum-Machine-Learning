function [psi_noisy, varargout] = Dephasing(psi, n, p)
% Dephasing applies n times dephasing noise to an m-qubit state psi
% with dephasing probability p per qubit
%
% in:
% psi: noiseless m-qubit state in tensor product basis as column vector
% n: number of noise realizations
% p: dephasing probability per qubit, maximal physical value is 0.5
% out:
% psi_noisy: psi_noisy(:,i) is the ith realization of a noisy psi
% varargout:
% {1}: n-element logical array v 
%      with v(i)= 1 if psi_noisy(:,i) experienced only 0 or 1 phase flips.

if p > 0.5
    error('Dephasing probability per qubit greater than 0.5 is unphysical.')
end

dim =  length(psi); % dimension of Hilbert space
m = log2(dim); % number of qubits

% define flips(:,:,i) as phase flip of ith spin
flip1 = [1, 0; 0, -1]; % dephasing for one qubit
id = eye(2);
flips = zeros(dim, dim, m);
for i = 1:m
    f = flip1;
    for j = 1:(i-1)
        f = kron(id, f);
    end
    for j = (i+1):m
        f = kron(f, id);
    end
    flips(:,:,i) = f;
end

% for every noise realization psi_noisy(:,i) and every qubit j
% draw random number r(j) from uniform distribution in (0, 1)
% and apply jth dephasing with probability p, i.e. if r(j) <= p 
psi_noisy = zeros(dim, n);
varargout = zeros(n,1);
for i = 1:n
    r = rand(m,1); % random numbers
    pn = psi;
    for j = 1:m
        if r(j) <= p
            pn = flips(:,:,j) * pn; % apply phase flip
        end
    end
    psi_noisy(:,i) = pn;
    nflips = (r <= p); 
    if sum(nflips) <= 1 % maximally one phase flip
        varargout(i) = 1;
    end
end
varargout = {logical(varargout)};
end
