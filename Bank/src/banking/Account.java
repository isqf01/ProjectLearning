package banking;

/**
 * a. 声明一个私有对象属性：balance，这个属性保留了银行帐户的当前（或 即时）余额。
 * b. 声 明 一 个 带 有一 个 参 数 （init_balance ） 的公 有 构 造 器 ，
 *    这个 参 数为balance 属性赋值。
 * c. 声明一个公有方法 geBalance，该方法用于获取经常余额。
 * d. 声明一个公有方法 deposit,该方法向当前余额增加金额。
 * e. 声明一个公有方法 withdraw 从当前余额中减去金额
 *
 */
public class Account {
    protected double balance;     //当前账户余额

    public Account(double init_balance){
        balance = init_balance;
    }

    //获取当前余额
    public double getBalance() {
        return balance;
    }

    //向当前余额增加余额
    public boolean deposit(double ant){
        balance += ant;
        return true;
    }

    public void withdraw(double ant) throws Exception {
        if (ant > balance){
            throw new OverdraftException("资金不足",ant-balance);
        }
        balance -= ant;
    }
}
