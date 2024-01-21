#Implementation of EOT PGD Algorithm

def adv_attack(model, X, y, device):
    X_adv = Variable(X.data)
    original_X = Variable(X.data)
    steps = 20
    eot_steps = 2
    epsilon = 0.1099
    alpha = 1e4

    delta = Variable(torch.zeros_like(original_X), requires_grad=True)

    randStart = True
    if randStart:
        random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-epsilon, epsilon).to(device)
        X_adv = X_adv + random_noise
        X_adv = Variable(torch.clamp(X_adv, 0, 1), requires_grad=True)

    for i in range(steps):
        data_grad = torch.zeros_like(X_adv)
        X_adv.requires_grad = True

        for ii in range(eot_steps):
            output = model(X_adv)

            loss = nn.CrossEntropyLoss()(output, y)

            loss.backward()
            data_grad += X_adv.grad.data

            sign_data_grad = torch.sign(data_grad)

            X_adv = X_adv + alpha * sign_data_grad

            delta = torch.clamp(X_adv - original_X, min=-epsilon, max=epsilon)
            X_adv = Variable(torch.clamp(original_X + delta, 0, 1), requires_grad=True)

            model.zero_grad()
    
    return X_adv
