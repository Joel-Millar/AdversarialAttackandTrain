Implementation of FGSM and PGD to train the model

def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0),28*28)

        epsilon = 0.1099
        #train with FGSM
        X_adv = Variable(data)
        y = target
        X_adv.requires_grad = True

        output = model(X_adv)

        loss = nn.CrossEntropyLoss()(output, y)

        loss.backward()
        data_grad = X_adv.grad.data

        sign_data_grad = torch.sign(data_grad)

        perturbed_image = X_adv + epsilon * sign_data_grad
        X_adv = torch.clamp(perturbed_image, 0, 1)
        adv_data = X_adv

        steps = 20
        alpha = 1e4
        X_adv = Variable(data)
        original_X = Variable(data)
        y = target
        randStart = True
        if randStart:
            random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-epsilon, epsilon).to(device)
            X_adv = X_adv + random_noise
            X_adv = torch.clamp(X_adv, 0, 1)

        X_adv.requires_grad = True
        output = model(X_adv)

        delta = torch.zeros_like(X_adv, requires_grad=True)

        for i in range(steps):
            output = model(X_adv)
            loss = nn.CrossEntropyLoss()(output, y)

            loss.backward()
            data_grad = X_adv.grad.data

            sign_data_grad = torch.sign(data_grad)

            helper = Variable(X_adv + alpha * sign_data_grad, requires_grad=True)

            # Clip(X_adv)
            delta.data = helper.clamp(min=-epsilon, max=epsilon)
            X_adv = Variable(torch.clamp(original_X + delta, 0, 1), requires_grad=True)
            model.zero_grad()

        adv_data2 = X_adv

        #clear gradients
        optimizer.zero_grad()
        
        #compute loss
        loss = F.nll_loss(model(data), target)
        loss1 = F.nll_loss(model(adv_data), target)
        loss2 = F.nll_loss(model(adv_data2), target)

        loss.backward()
        loss1.backward()
        loss2.backward()

        optimizer.step()
