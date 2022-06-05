package banking;

/**
 * 1.CheckingAccount 类必须扩展 Account 类
 * 2.该类必须包含一个类型为 double 的 overdraftProtection 属性。
 * 3.该类必须包含一个带有参数（balance）的共有构造器。
 *  该构造器必须通过调用 super(balance)将 balance 参数传递给父类构造器。
 * 4.给类必须包括另一个带有两个参数（balance 和 protect）的公有构造器。
 *  该构造器必须通过调用 super(balance)并设置 overdragtProtection 属性，将 balance 参数传递给父类构造器。
 * 5.CheckingAccount 类必须覆盖 withdraw 方法。此方法必须执行下列检查。
 *  如果当前余额足够弥补取款 amount,则正常进行。如果不够弥补但是存在透支保护，
 *  则尝试用 overdraftProtection 得值来弥补该差值（balance-amount）.
 *  如果弥补该透支所需要的金额大于当前的保护级别。则整个交易失败，但余额未受影响。
 */
public class CheckingAccount extends Account {
    private SavingAccount protectedBy;
    private double protectedBy1 = -1;

    public CheckingAccount(double init_balance) {
        super(init_balance);
    }

    public CheckingAccount(double init_balance, SavingAccount protectedBy) {
        super(init_balance);
        this.protectedBy = protectedBy;
    }
    public CheckingAccount(double init_balance, double protectedBy1) {
        super(init_balance);
        this.protectedBy1 = protectedBy1;
    }
    @Override
    public void withdraw(double ant) throws Exception {
        if (balance >= ant){    //余额足够
            balance -= ant;
        }else {
            if (protectedBy1 == -1){
                throw new OverdraftException("no overdraft protection",ant-balance);
            }
            //透支保护足够
            if (protectedBy!=null && protectedBy.getBalance() >=(ant - balance)){
                protectedBy.withdraw(ant - balance);
                balance = 0;
            }
            //余额不够，且透支保护也不够
            else {
                throw new OverdraftException("Insufficient funds for overdraft protection",ant-balance);
            }
        }

    }

}
