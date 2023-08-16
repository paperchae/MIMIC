import numpy as np
import matplotlib.pyplot as plt

def hrv_report(time, index_list, freq, info):
    for i in range(len(info)):
        hr = time.hr[i]
        sdnn = time.sdnn[i]
        rmssd = time.rmssd[i]
        nn50 = time.nn50[i]
        pnn50 = time.pnn50[i]
        # apen = time.apen[i]
        srd = time.srd[i]
        t_power = freq.t_power[i]
        vlf = freq.vlf_power[i]
        lf = freq.lf_power[i]
        hf = freq.hf_power[i]
        normalized_lf = freq.norm_lf[i]
        normalized_hf = freq.norm_hf[i]
        lf_hf_ratio = freq.lf_hf[i]
        fig, ax = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [2, 1]}, figsize=(10, 6))
        fig.suptitle('MIMIC-III HRV Report \n\nSubject ID: {} Hospital Admin ID: {} Age: {} Gender: {} Diagnosis: {}'.format(
            info[i][0], info[i][1], info[i][2], info[i][3], info[i][5]), fontsize=11, fontweight='bold')
        ax[0, 0].set_title('HRV Tachogram', fontsize=9, fontweight='bold')
        ax[0, 0].set_xlabel('Time (min)')
        ax[0, 0].set_ylabel('HR (bpm)')
        ax[0, 0].set_ylim(40, 140)
        ax[0, 0].text(0, 130,
                      'mean RRI: ' + str(np.round(np.mean(time.input_signals[i][time.input_signals[i] > 0].numpy()),
                                                  1)) + '(ms), mean HR:' + str(
                          np.round(np.mean(hr.numpy()), 1)) + '(bpm)')
        # ax[0, 0].plot(time.input_signals[i][time.input_signals[i] > 0], color='royalblue',
        #               label='Input Signal')
        t = (index_list[i][index_list[i] > 0] / 18000) * 5
        t = t[:len(time.input_signals[i][time.input_signals[i] > 0])]
        hr_tacho = 60 / (time.input_signals[i][time.input_signals[i] > 0] / 1000)
        max_hr = np.max(hr_tacho.numpy())
        min_hr = np.min(hr_tacho.numpy())
        ax[0, 0].fill_between(t, np.ones(len(t)) * 60, np.ones(len(t)) * 100, color='lightgray', alpha=0.5)
        ax[0, 0].plot(t, hr_tacho, color='royalblue')
        ax[0, 0].axhline(y=max_hr, xmin=0.05, xmax=0.95, color='orange', linestyle='--', linewidth=0.9, label='Max HR')
        ax[0, 0].axhline(y=min_hr, xmin=0.05, xmax=0.95, color='orange', linestyle='-.', linewidth=0.9, label='Min HR')
        # ax[0, 0].text(0, max_hr + 2, str(np.round(max_hr, 2)))
        # ax[0, 0].text(0, min_hr - 7, str(np.round(min_hr, 2)))
        ax[0, 0].legend(loc='upper right', fontsize='small')
        bar_plot_x = np.arange(2)
        bar_plot_x_values = [sdnn, rmssd]
        ax[0, 1].set_title('Time Domain Components', fontsize=9, fontweight='bold')
        ax[0, 1].bar(bar_plot_x, bar_plot_x_values,
                     color=['mediumseagreen', 'darkorange'], width=0.4)
        ax[0, 1].set_xticks(bar_plot_x, ['SDNN', 'RMSSD'])
        ax[0, 1].hlines(y=30, xmin=-0.1, xmax=0.1, color='gray')
        ax[0, 1].hlines(y=50, xmin=-0.1, xmax=0.1, color='gray')
        ax[0, 1].hlines(y=20, xmin=0.9, xmax=1.1, color='gray')
        ax[0, 1].hlines(y=40, xmin=0.9, xmax=1.1, color='gray')
        ax[0, 1].vlines(x=0, ymin=30, ymax=50, color='gray')
        ax[0, 1].vlines(x=1, ymin=20, ymax=40, color='gray')
        ax[0, 1].hlines(y=20, xmin=-0.1, xmax=0.1, color='red')
        ax[0, 1].hlines(y=10, xmin=0.9, xmax=1.1, color='red')
        ax[0, 1].text(0, sdnn, str(np.round(sdnn.numpy(), 3)), ha='center', va='bottom')
        ax[0, 1].text(1, rmssd, str(np.round(rmssd.numpy(), 3)), ha='center', va='bottom')
        # ax[0, 1].axes.yaxis.set_visible(False)
        ax[1, 0].set_title('Signal Decomposition', fontsize=9, fontweight='bold')
        sig_t = np.arange(0, 5, 5 / 18000)
        ax[1, 0].plot(sig_t, freq.hf_signal[i], color='darkorange', label='HF: 0.15~0.4 Hz')
        ax[1, 0].plot(sig_t, freq.lf_signal[i], color='mediumseagreen', label='LF: 0.04~0.15 Hz')
        ax[1, 0].plot(sig_t, freq.vlf_signal[i], color='royalblue', label='VLF: 0.003~0.04 Hz')
        ax[1, 0].set_xlabel('Time (min)')
        ax[1, 0].legend(loc='upper right', fontsize='small')
        ax[1, 0].axes.yaxis.set_visible(False)

        ax[1, 1].set_title('Frequency Domain Components', fontsize=9, fontweight='bold')
        bar_plot_x = np.arange(3)
        bar_plot_x_values = [vlf, lf, hf]
        ax[1, 1].bar(bar_plot_x, bar_plot_x_values,
                     color=['royalblue', 'mediumseagreen', 'darkorange'], width=0.4)
        ax[1, 1].set_xticks(bar_plot_x, ['VLF', 'LF', 'HF'])
        ax[1, 1].set_ylim(0, np.max(bar_plot_x_values) * 1.3)
        ax[1, 1].text(1, lf, 'LF Norm\n' + str(np.round(normalized_lf.numpy(), 3)), ha='center', va='bottom')
        ax[1, 1].text(2, hf, 'HF Norm\n' + str(np.round(normalized_hf.numpy(), 3)), ha='center', va='bottom')
        ax[1, 1].text(1.5, (lf + hf) / 2, 'LF/HF\n' + str(np.round(lf_hf_ratio.numpy(), 2)), ha='center', va='bottom')
        ax[1, 1].axes.yaxis.set_visible(False)
        fig.tight_layout()
        # plt.show()
        plt.savefig('./hrv_img/'+str('Subject ID: {} Hospital Admin ID: {} Age: {} Gender: {} Diagnosis: {}'.format(
            info[i][0], info[i][1], info[i][2], info[i][3], info[i][5]))+'.png')
        plt.close()


def hrv_comparison_plot(raw_tar, raw_tar_hrv, raw_tar_index, f_tar, f_tar_hrv, f_tar_index,
                        raw_pred, raw_pred_hrv, raw_pred_index, f_pred, f_pred_hrv, f_pred_index):
    # target
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))
    fig.suptitle('Target BPV signal HRV Analysis \n', fontweight='bold')
    ax[0].set_title('(a)')
    ax[0].plot(raw_tar.cpu().numpy(), '-.', color='gray', label='raw target')
    ax[0].plot(raw_tar_index[raw_tar_index >= 0].cpu().numpy(),
               raw_tar[raw_tar_index[raw_tar_index >= 0].cpu().numpy()].cpu().numpy(), 'b.',
               label='raw target peak')
    ax[0].plot(f_tar.cpu().numpy(), color='darkorange', label='filtered target')
    ax[0].plot(f_tar_index[f_tar_index >= 0].cpu().numpy(),
               f_tar[f_tar_index[f_tar_index >= 0].cpu().numpy()].cpu().numpy(), 'rx',
               label='filtered target peak')
    ax[0].set_xticks(np.arange(0, len(raw_tar) + 30, 30), np.arange(0, len(raw_tar) + 30, 30) // 30)
    ax[0].set_xlabel('Time (seconds)')
    ax[0].legend(loc='lower right')
    # prediction
    ax[1].set_title('(b)')
    ax[1].plot(raw_tar_hrv[raw_tar_hrv >= 0].cpu().numpy(), '-.', color='gray', label='raw target hrv')
    ax[1].plot(raw_tar_hrv[raw_tar_hrv >= 0].cpu().numpy(), 'b.')
    ax[1].plot(f_tar_hrv[f_tar_hrv >= 0].cpu().numpy(), color='darkorange', label='filtered target hrv')
    ax[1].plot(f_tar_hrv[f_tar_hrv >= 0].cpu().numpy(), 'rx')
    ax[1].set_xlabel('HRV Count')
    ax[1].set_xticks(np.arange(0, len(raw_tar_hrv[raw_tar_hrv >= 0]), 1))
    ax[1].set_ylabel('HRV (milliseconds)')
    ax[1].legend(loc='lower right')
    fig.tight_layout()
    plt.show()
    plt.close()

    # prediction
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))
    fig.suptitle('Predicted BVP signal HRV Analysis \n', fontweight='bold')
    ax[0].set_title('(a)')  # raw BVP HR: ' + str(int(raw_pred_hr)) + ' filtered BVP HR: ' + str(int(f_pred_hr)))
    ax[0].plot(raw_pred.cpu().numpy(), '-.', color='gray', label='raw prediction')
    ax[0].plot(raw_pred_index[raw_pred_index >= 0].cpu().numpy(),
               raw_pred[raw_pred_index[raw_pred_index >= 0].cpu().numpy()].cpu().numpy(), 'b.',
               label='raw prediction peak')
    ax[0].plot(f_pred.cpu().numpy(), color='darkorange', label='filtered prediction')
    ax[0].plot(f_pred_index[f_pred_index >= 0].cpu().numpy(),
               f_pred[f_pred_index[f_pred_index >= 0].cpu().numpy()].cpu().numpy(), 'rx',
               label='filtered prediction peak')
    ax[0].set_xticks(np.arange(0, len(raw_tar) + 30, 30), np.arange(0, len(raw_tar) + 30, 30) // 30)
    ax[0].set_xlabel('Time (seconds)')
    ax[0].legend(loc='lower right')
    # plt.subplot(2, 1, 2)
    ax[1].set_title('(b)')  # HRV Comparison' + '[ Peak score: ' + str(peak_score) + ']')
    ax[1].plot(raw_pred_hrv[raw_pred_hrv >= 0].cpu().numpy(), '-.', color='gray', label='raw prediction hrv')
    ax[1].plot(raw_pred_hrv[raw_pred_hrv >= 0].cpu().numpy(), 'b.')
    ax[1].plot(f_pred_hrv[f_pred_hrv >= 0].cpu().numpy(), color='darkorange', label='filtered prediction hrv')
    ax[1].plot(f_pred_hrv[f_pred_hrv >= 0].cpu().numpy(), 'rx')
    ax[1].set_xlabel('HRV Count')
    ax[1].set_xticks(np.arange(0, len(raw_pred_hrv[raw_pred_hrv >= 0]), 1))
    ax[1].set_ylabel('HRV (milliseconds)')
    plt.legend(loc='lower right')
    fig.tight_layout()
    plt.show()


def hr_comparison_bpf(hr_label_fft, hr_pred_fft, hr_pred_fft_filtered,
                      hr_label_peak, hr_pred_peak, hr_pred_peak_filtered):
    plt.title('(a)')
    plt.scatter(x=hr_label_fft.detach().cpu().numpy(),
                y=hr_pred_fft.detach().cpu().numpy(),
                color='blue', alpha=0.2, marker='2', label='FFT HR Prediction')
    plt.scatter(x=hr_label_fft.detach().cpu().numpy(),
                y=hr_pred_fft_filtered.detach().cpu().numpy(),
                color='red', alpha=0.2, marker='1', label='FFT HR Prediction Filtered')
    plt.xlim(40, 150)
    plt.xlabel('Target HR')
    plt.ylabel('Predicted HR')
    plt.legend(loc='upper left')
    plt.show()
    plt.title('(b)')
    plt.scatter(x=hr_label_peak[0].detach().cpu().numpy(),
                y=hr_pred_peak[0].detach().cpu().numpy(),
                color='blue', alpha=0.2, marker='2', label='Peak HR Prediction')
    plt.scatter(x=hr_label_peak[0].detach().cpu().numpy(),
                y=hr_pred_peak_filtered.detach().cpu().numpy(),
                color='red', alpha=0.2, marker='1', label='Peak HR Prediction Filtered')
    plt.legend(loc='upper left')
    plt.xlim(40, 150)
    plt.xlabel('Target HR')
    plt.ylabel('Predicted HR')
    plt.show()
