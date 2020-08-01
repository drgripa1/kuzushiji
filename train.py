import os

import data
import models
import options


def train(opt):
    print(opt)
    with open(os.path.join(opt.checkpoint_dir, 'loss_log.txt'), 'a') as f:
        f.write(str(opt) + '\n')
    dataloader = data.get_dataloader(True, opt.batch, opt.dataset_dir)
    model = models.ResNetModel(opt, train=True)

    total_iter = 0
    loss = 0.0

    while True:
        for batch in dataloader:
            total_iter += 1
            inputs, labels = batch
            model.optimize_params(inputs, labels)
            loss += model.get_current_loss()

            if total_iter % opt.print_freq == 0:
                txt = f'iter: {total_iter: 6d}, loss: {loss / opt.print_freq}'
                print(txt)
                with open(os.path.join(opt.checkpoint_dir, 'loss_log.txt'), 'a') as f:
                    f.write(txt + '\n')
                loss = 0.0

            if total_iter % opt.save_params_freq == 0:
                model.save_model(f'{total_iter // opt.save_params_freq}k')

            if total_iter == opt.n_iter:
                model.save_model('final')
                return


if __name__ == '__main__':
    args = options.parse_args_train()
    train(args)
