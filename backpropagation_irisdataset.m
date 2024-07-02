clc 
clear

% Import dataset
data_name = 'Iris.csv';
data = readtable(data_name);

data = removevars(data, {'Id'});

% Mencari informasi umum dari dataset
column_names = data.Properties.VariableNames; % mengetahui nama setiap kolom dari data
row_num = size(data, 1); % mengetahui jumlah baris dari data
column_num = size(data,2); % mengetahui jumlah kolom dari data


% One Hot Encoding
i = 1;
while i <= column_num % mengecek setiap kolom
    if iscell(data.(column_names{i})) % mengecek apakah kolom mengandung data teks
        
        keys = unique(data.(column_names{i}))'; % Mendapatkan variabel unique dari satu kolom
        values = cell(1, size(keys, 2));

        % Mengubah variabel unique ke dalam bentuk angka
        for j = 1:size(keys, 2) % va
            values{j} = j; 
        end

        % Membuat mapping dari angka ke teks unique
        map = containers.Map(keys, values);
        for j = 1:row_num
            data.(column_names{i}){j} = map(data.(column_names{i}){j});
        end
        data.(column_names{i}) = cell2mat(data.(column_names{i}));
        
        %one hot encoding
        tmp = double(data.(column_names{i}) == 1:size(values,2));
        table_name = cell(1, size(tmp, 2));
        for j = 1:size(tmp, 2)
            table_name{j} = strcat((column_names{i}), num2str(j));
        end
        
        tmp = array2table(tmp, 'VariableNames', table_name);

        % sisipkan one-hot encoding tersebut dan menghapus yang lama
        data = [data(:, 1:i-1) tmp data(:, i+1:end)];
        i = size([data(:, 1:i-1) tmp], 2) + 1;
        column_names = data.Properties.VariableNames;
    else
        i = i + 1;
    end
end


% Pemisahan data
data_random = data(randperm(size(data,1)), :);

% Memisahkan feature dan target
feature = data_random(:, 1:end-3);
target = data_random(:, 5:end);

feature_num_row = size(feature, 1);
feature_num_column = size(feature, 2);

target_num_row = size(target, 1);
target_num_column = size(target, 2);


% Pemisahan Data Training dengan Data Testing
train_size = 0.7 * feature_num_row;

X_train = table2array(feature(1:train_size, :));
y_train = table2array(target(1:train_size, :));

X_test = table2array(feature(train_size + 1:end, :));
y_test = table2array(target(train_size + 1:end, :));



% Melakukan Input Layer, Hidden Layer, dan Output Layer
n = feature_num_column; % input layer
p = 5;                 % hidden layer
m = target_num_column;  % output layer


% Penentuan nilai bobot menggunakan metode nguyen widrow
a = -0.5;
b = 0.5;

V = rand(n, p) + a;
W = rand(p, m) - b;

beta_V = 0.7 * (p) .^ (1/n);
beta_W = 0.7 * (m) .^ (1/p);

V_0j = -beta_V + (beta_V - (-beta_V)) .* rand(1,p);
W_0k = -beta_W + (beta_W - (-beta_W) .* rand(1,m));

V_j = sqrt(sum(V.^2));
W_k = sqrt(sum(W.^2));

Vij_new = (beta_V .* V) / V_j;
Wjk_new = (beta_W .* W) / W_k;


% Inisialisasi nilai lama
W_jk_lama = 0;
W_0k_lama = 0;
V_ij_lama = 0;
V_0j_lama = 0;


% Penentuan nilai parameter iterasi
iterasi = 200;
iter = 0;
Ep_stop = 1;
alpha = 0.2;
miu = 0.4;


% Training data
while Ep_stop == 1 && iter < iterasi
    iter = iter + 1;
    for a = 1:length(X_train)
        % Proses feedforward
        z_inj = V_0j + X_train(a,:) * V;
        
        % Proses aktivasi menggunakan fungsi sigmoid
        for j = 1:p
            zj(1,j) = 1/(1+exp(-z_inj(1,j)));
        end
        
        y_ink = W_0k + zj * W;
        for r = 1:m
            yk(1,r) = 1/(1+exp(-y_ink(1,r))); % Aktivasi sigmoid
        end
        
        % Menghitung nilai error
        E(1, a) = sum(abs(y_train(a,:) - yk));
        
        % Proses backpropagation
        do_k = (y_train(a,:) - yk) .* (yk .* (1-yk));
        W_jk = alpha * zj' * do_k + miu * W_jk_lama;
        W_0k = alpha * do_k + miu * W_0k_lama;
        
        W_jk_lama = W_jk;
        W_0k_lama = W_0k;
        
        do_inj = do_k * W';
        do_j = do_inj .* (zj .* (1-zj));
        V_ij = alpha * X_train(a,:)' * do_j + miu * V_ij_lama;
        V_0j = alpha * do_j + miu * V_0j_lama;
        
        V_ij_lama = V_ij;
        V_0j_lama = V_0j;
        
        W = W + W_jk;
        W_0k = W_0k + W_0k;
        
        V = V + V_ij;
        V_0j = V_0j + V_0j;
    end
    
    % Menghitung nilai error pada tiap epoch
    Ep(1, iter) = sum(E) / length(X_train);
    
    if Ep(1,iter) < 0.01
        Ep_stop = 0;
    end
    acc_p(1,iter) = 1 - Ep(1, iter);
end

% Melakukan testing
E_test = zeros(1,length(X_test));
right = 0;
wrong = 0;

for a = 1:length(X_test)
    z_inj_test = X_test(a,:)*V + V_0j;
    % Proses aktivasi menggunakan sigmoid
    
    for j=1:p
        zj_test(1,j) = 1/(1+exp(-z_inj_test(1,j))); %Aktivasi sigmoid
    end
    
    y_ink_test = zj_test * W + W_0k;
    
    for k=1:m
        yk_test(1,k) = 1/(1+exp(-y_ink_test(1,k))); %Aktivasi sigmoid
    end
    
    for j = 1:m
        predict(a,j) = yk_test(j);
    end
    
    %Menghitung nilai error
    E_test(1,a) = sum(abs(y_test(a,:) - yk_test));
    
    [value, index] = max(yk_test);
    Y_test = zeros(train_size, target_num_column);
    Y_test(a, index) = 1;

    if Y_test(a, :) == y_test(a,:) % Menghitung jumlah prediksi benar
        right = right + 1;

    else % Menghitung jumlah prediksi salah
        wrong = wrong + 1;
    end
end

% Evaluasi hasil data testing
avgerrortest = sum(E_test)/length(X_test);
recog_rate = (right/length(X_test))*100;

accuracy = right/(right+wrong);
error_rate = wrong/(right+wrong);

figure;
plot(Ep);
ylabel('Error'); xlabel('Iterasi')

figure;
plot(acc_p);
ylabel('Accuracy'); xlabel('Iterasi')

disp("Didapat nilai error " + Ep(end) + " pada iterasi ke-" + iter);
disp("Accuracy rate = " + (accuracy*100) + "%");
disp("Error rate    =  " + (error_rate*100) + "%");
disp("Accuracy rate pada training sebesar " + (acc_p(end)*100) + "%");
disp(" ");
disp("Pada test diperoleh: ");
disp("Error sebesar " + E_test(end) + " pada iterasi ke-" + iter);
disp("Jumlah Benar  = " + right + "/" + (right+wrong));