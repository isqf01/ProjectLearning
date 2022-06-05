package banking;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * a. 声明三个私有对象属性：firstName、lastName 和 account。
 * b. 声明一个公有构造器，这个构造器带有两个代表对象属性的参数（f 和 l）
 * c. 声明两个公有存取器来访问该对象属性，方法 getFirstName 和 getLastName返回相应的属性。
 * d. 声明 setAccount 方法来对 account 属性赋值。
 * e. 声明 getAccount 方法以获取 account 属性。
 */
public class Customer {
    private String firstName;
    private String lastName;
//    private Account[] accounts;
//    private int NumberOfAccounts;
    private List<Account> accounts;

    private SavingAccount savingAccount;
    private CheckingAccount checkingAccount;

    public Customer(String f, String l) {
        this.firstName = f;
        this.lastName = l;
//        accounts = new Account[5];
        accounts = new ArrayList<>();
    }

    public String getFirstName() {
        return firstName;
    }

    public String getLastName() {
        return lastName;
    }

   public Account getAccount(int index) {
//        return accounts[index];
       return accounts.get(index);
    }

    public void addAccount(Account acc) {
//        this.accounts[NumberOfAccounts++] = acc;
        accounts.add(acc);
    }

    public int getNumOfAccounts() {
//        return NumberOfAccounts;
    return accounts.size();
    }

    public SavingAccount getSavingAccount() {
        return savingAccount;
    }

    public CheckingAccount getCheckingAccount() {
        return checkingAccount;
    }

    public void setSavingAccount(SavingAccount savingAccount) {
        this.savingAccount = savingAccount;
    }

    public void setCheckingAccount(CheckingAccount checkingAccount) {
        this.checkingAccount = checkingAccount;
    }
    public Iterator<Account> getAccount(){
        return accounts.iterator();
    }
}
